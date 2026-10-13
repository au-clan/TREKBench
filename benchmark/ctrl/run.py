#!/usr/bin/env python3
"""Run the frozen Trek-Ctrl experiment without the original repository."""
import argparse
import gzip
import hashlib
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .api import Client, fingerprint, safe_error, save_json
from .methods import CONTEXTS, HERE, METHODS, answer, grade
from benchmark.data import DATA, ROOT, load_records

DATASETS = ("freshqa", "infodeepseek", "sealqa")


def load_data(dataset):
    return [{**row['ctrl'], 'id': row['id']} for row in load_records() if row['dataset'] == dataset]


def expand(values, allowed):
    if values == ["all"]:
        return list(allowed)
    if not values or any(value not in allowed for value in values):
        raise ValueError(f"Choose 'all' or one or more of: {', '.join(allowed)}")
    return list(dict.fromkeys(values))


def build_jobs(datasets, methods, conditions, limit):
    jobs = []
    for dataset in datasets:
        tasks = load_data(dataset)
        for condition in conditions:
            # Gold Only is a shared direct-generation reference, not four copies.
            for method in (["direct"] if condition == "gold" else methods):
                for task in (tasks[:limit] if limit else tasks):
                    jobs.append((dataset, method, condition, task))
    return jobs


def identity(job):
    dataset, method, condition, task = job
    return {"dataset": dataset, "method": method, "condition": condition, "task_id": task["id"]}


def case_path(output, job):
    return output / "cases" / (fingerprint(identity(job)) + ".json")


def ensure_run(output, spec):
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError("Output folder has different data/code/config/jobs. Use a new --out folder.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output folder without a manifest. Use a new --out folder.")
    else:
        save_json(path, spec)


def execute(job, output, answer_client, judge_client, do_judge, reranker_config=None):
    dataset, method, condition, task = job
    path = case_path(output, job)
    row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        **identity(job), "question": task["question"], "references": task["answers"]}
    row.pop("error", None)
    try:
        if "prediction" not in row:
            prediction, trace = answer(answer_client, task, method, condition, reranker_config)
            row.update(prediction=prediction, trace=trace, status="generated")
            save_json(path, row)  # Preserve generation even if judging later fails.
        if do_judge and "grade" not in row:
            row["grade"] = grade(judge_client, dataset, task, row["prediction"])
        row["status"] = "scored" if "grade" in row else "generated"
    except Exception as error:
        row.update(status="judge_error" if "prediction" in row else "generation_error",
                   error=safe_error(error))
    save_json(path, row)
    return row


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['dataset']}/{row['method']}/{row['condition']}"].append(row)
        groups[f"all/{row['method']}/{row['condition']}"].append(row)
    result = {}
    for name, items in sorted(groups.items()):
        scored = [row for row in items if "grade" in row]
        correct = sum(row["grade"]["correct"] is True for row in scored)
        result[name] = {
            "expected": len(items), "generated": sum("prediction" in row for row in items),
            "scored": len(scored), "correct": correct if scored else None,
            "errors": sum("error" in row for row in items),
            "accuracy_percent": 100 * correct / len(items) if len(scored) == len(items) else None,
            "accuracy_on_scored_percent": 100 * correct / len(scored) if scored else None,
        }
        recalls = [row["trace"]["gold_document_recall_at_k"] for row in items
                   if "gold_document_recall_at_k" in row.get("trace", {})]
        if recalls:
            result[name]["mean_gold_document_recall_at_k"] = {
                str(k): sum(values) / len(values) if values else None
                for k in (1, 3, 5, 10)
                for values in [[r[str(k)] for r in recalls if r[str(k)] is not None]]}
    for name in list(result):
        if name.startswith('all/'):
            suffix = name[3:]
            values = [result[ds + suffix]['accuracy_percent'] for ds in DATASETS if ds + suffix in result]
            result[name]['micro_accuracy_percent'] = result[name]['accuracy_percent']
            result[name]['macro_accuracy_percent'] = (sum(values) / len(values)
                if values and all(v is not None for v in values) else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--datasets", nargs="+", default=["freshqa"])
    parser.add_argument("--methods", nargs="+", default=["direct"])
    parser.add_argument("--conditions", nargs="+", default=["NMD"])
    parser.add_argument("--limit", type=int, default=1, help="Questions per dataset; 0 = all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--judge", action="store_true", help="Also score against supplied references")
    parser.add_argument("--dry-run", action="store_true", help="Check data/jobs without any API calls")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "ctrl")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "ctrl")
    args = parser.parse_args()
    try:
        if args.limit < 0 or args.workers < 1:
            raise ValueError("--limit must be >= 0 and --workers must be >= 1")
        datasets = expand(args.datasets, DATASETS)
        methods = expand(args.methods, METHODS)
        conditions = expand(args.conditions, CONTEXTS)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        for section in ("answer", "judge"):
            item = config[section]
            if not all(item.get(key) for key in ("base_url", "api_key_env", "model")):
                raise ValueError(f"Missing endpoint, key environment name, or model in {section}")
            if "REPLACE" in item["model"]:
                raise ValueError(f"Set an actual model ID in config.{section}.model")
        jobs = build_jobs(datasets, methods, conditions, args.limit)
        cells = {tuple(identity(job)[k] for k in ("dataset", "method", "condition")) for job in jobs}
        print(f"{len(jobs)} answers across {len(cells)} cells; judge={'on' if args.judge else 'off'}", flush=True)
        if args.dry_run:
            print("Dry run passed: data hashes verified; no API calls or scores produced.")
            return 0
        files = [HERE / name for name in ("api.py", "methods.py", "run.py")]
        files += [HERE / 'rerank.py', ROOT / 'benchmark/data.py']
        files += sorted((HERE.parent / "prompts").glob("*.md")) + [DATA / "manifest.json"]
        spec = {"config": config, "jobs": [identity(job) for job in jobs],
                "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        ensure_run(args.out, spec)
        stop = threading.Event()
        answer_client, judge_client = (Client(config[key], args.cache, stop) for key in ("answer", "judge"))
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            reranker = {**config.get('reranker', {}), 'cache': str(args.cache / 'rerank')}
            futures = [pool.submit(execute, job, args.out, answer_client, judge_client, args.judge, reranker) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"[{len(rows)}/{len(jobs)}] {row['dataset']} {row['task_id']} "
                      f"{row['method']}/{row['condition']}: {row['status']}", flush=True)
        summary = summarize(rows)
        save_json(args.out / "summary.json", summary)
        print(f"Saved {args.out / 'summary.json'} (null accuracy means scoring is incomplete).")
        return int(any("error" in row for row in rows))
    except (ValueError, KeyError, OSError) as error:
        parser.error(safe_error(error))


if __name__ == "__main__":
    raise SystemExit(main())
