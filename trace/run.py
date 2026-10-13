"""Diagnose failed Wild predictions, using the original model for both probes."""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark.ctrl.api import fingerprint, save_json, safe_error
from benchmark.data import DATA, ROOT
from benchmark.wild.common import ensure_run, latest_records, load_tasks, make_client
from benchmark.wild.run import extract_trace
from .core import diagnose


class ProbeClient:
    def __init__(self, client, model, effort, cache):
        self.client, self.model, self.effort, self.cache = client, model, effort, Path(cache)

    def call(self, prompt, search=False):
        request = {'model': self.model, 'reasoning': {'effort': self.effort}, 'input': prompt}
        if search:
            request.update(tools=[{'type': 'web_search'}], include=['web_search_call.action.sources'])
        key = fingerprint({'request': request, 'base_url': str(self.client.base_url)})
        path = self.cache / (key + '.json')
        if path.exists():
            return json.loads(path.read_text())
        response = self.client.responses.create(**request)
        raw = response.model_dump(mode='json')
        if raw.get('status') != 'completed' or not response.output_text.strip():
            raise ValueError('Empty/incomplete diagnostic response')
        result = {'request_id': key, 'text': response.output_text, 'usage': raw.get('usage'),
                  **extract_trace(raw)}
        save_json(path, result)
        return result

    def forget(self, response):
        key = response.get('request_id', '')
        if len(key) == 64 and all(c in '0123456789abcdef' for c in key):
            (self.cache / (key + '.json')).unlink(missing_ok=True)


def make_case(dataset, task, prediction):
    queries = []
    for call in prediction.get('search_calls', []):
        action = call.get('action') or call
        values = action.get('queries') or ([action['query']] if action.get('query') else [])
        queries.extend(values if isinstance(values, list) else [values])
    return {'dataset': dataset, 'task_id': task['id'], 'question': task['question'],
            'effective_year': task.get('metadata', {}).get('effective_year'),
            'gold_answers': task.get('answers') or [task['answer']],
            'gold_urls': list(dict.fromkeys(u.split('#', 1)[0] for u in task['gold_urls'])),
            'prediction': prediction['prediction'], 'search_queries': list(dict.fromkeys(queries)),
            'retrieved_urls': prediction['retrieved_urls']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--grades', type=Path, required=True)
    parser.add_argument('--human-labels', type=Path, help='JSON object: task_id -> verified system-unrelated category')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--judge-model', default='gpt-5.4-mini')
    parser.add_argument('--judge-reasoning', default='low')
    parser.add_argument('--audit-model', default='gpt-5.4')
    parser.add_argument('--audit-reasoning', default='high')
    parser.add_argument('--workers', type=int, default=2, help='Each case has two independent branches')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/trace')
    parser.add_argument('--cache', type=Path, default=ROOT / '.cache/trace')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1:
        parser.error('limit >= 0 and workers >= 1 required')
    predictions, grades = latest_records(args.predictions), latest_records(args.grades)
    tasks = load_tasks(args.data_dir)
    human = json.loads(args.human_labels.read_text()) if args.human_labels else {}
    if not isinstance(human, dict) or any(not isinstance(v, str) or not v.strip() for v in human.values()):
        parser.error('Human labels must map task IDs to nonempty system-unrelated labels')
    jobs = []
    for key, graded in grades.items():
        if graded.get('status') != 'ok' or graded.get('correct') is not False:
            continue
        prediction = predictions.get(key)
        if key not in tasks or not prediction or prediction.get('status') != 'ok':
            parser.error(f'Missing valid prediction/task: {key}')
        if prediction['prediction'] != graded.get('prediction'):
            parser.error(f'Grade belongs to a different prediction: {key}')
        if 'retrieved_urls' not in prediction or 'search_calls' not in prediction:
            parser.error(f'Prediction has no complete search trace: {key}')
        if not prediction.get('search_enabled'):
            parser.error('Trace expects search-enabled Wild runs, not closed-book predictions')
        if not prediction.get('model') or not prediction.get('reasoning'):
            parser.error('Original model and reasoning settings must be recorded')
        jobs.append((make_case(key[0], tasks[key], prediction), prediction['model'], prediction['reasoning']))
    jobs = jobs[:args.limit or None]
    print(f'{len(jobs)} failures selected; {sum(c["task_id"] in human for c, _, _ in jobs)} human exclusions')
    if args.dry_run or not jobs:
        return 0
    files = list((ROOT / 'trace').glob('*.py')) + list((ROOT / 'trace/prompts').glob('*.md'))
    spec = {'jobs': fingerprint(jobs), 'human_labels': human, 'audit': [args.audit_model, args.audit_reasoning],
            'judge': [args.judge_model, args.judge_reasoning],
            'code': fingerprint([p.read_text() for p in sorted(files)])}
    ensure_run(args.out, spec)
    sdk = None if all(case['task_id'] in human for case, _, _ in jobs) else make_client()
    auditor = ProbeClient(sdk, args.audit_model, args.audit_reasoning, args.cache)
    grader = ProbeClient(sdk, args.judge_model, args.judge_reasoning, args.cache)

    def execute(job):
        case, model, effort = job
        path = args.out / 'cases' / (fingerprint([case['dataset'], case['task_id']]) + '.json')
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get('status') == 'ok':
                return previous
        try:
            row = diagnose(case, ProbeClient(sdk, model, effort, args.cache), auditor, grader,
                           human.get(case['task_id']))
            row['test_model'] = model
            row['test_reasoning'] = effort
        except Exception as exc:
            row = {**case, 'status': 'error', 'error': safe_error(exc)}
        save_json(path, row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            print(row['task_id'], row['status'], flush=True)
    rows.sort(key=lambda r: (r['dataset'], r['task_id']))
    (args.out / 'diagnoses.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    counts = Counter(c for r in rows for c in r.get('diagnosis', {}).get('system_categories', []))
    save_json(args.out / 'summary.json', {'cases': len(rows), 'errors': sum(r['status'] != 'ok' for r in rows),
        'system_unrelated': sum(r.get('diagnosis', {}).get('primary_category') == 'system-unrelated' for r in rows),
        'multi_label_counts': dict(counts)})
    return int(any(r['status'] != 'ok' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
