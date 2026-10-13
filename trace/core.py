"""Two independent diagnostic branches, with inspectable evidence."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

PROMPTS = Path(__file__).resolve().parent / 'prompts'
FLAGS = ('pre_retrieval', 'retrieval', 'post_retrieval_distraction',
         'post_retrieval_interpretation', 'randomness')


def identity(url):
    parsed = urlsplit(str(url).strip())
    return urlunsplit(('', parsed.netloc.lower().removeprefix('www.'),
                       parsed.path.rstrip('/'), parsed.query, ''))


def prompt(name, case, **extra):
    values = {**case, 'effective_year': case.get('effective_year') or '(unspecified)',
              'gold_answers': ' | '.join(case['gold_answers']), **extra}
    for key in ('gold_urls', 'retrieved_urls', 'search_queries', 'context_urls'):
        if key in values and isinstance(values[key], list):
            values[key] = '\n'.join('- ' + x for x in values[key]) or '(none)'
    text = (PROMPTS / (name + '.md')).read_text(encoding='utf-8')
    for key, value in values.items():
        text = text.replace('{' + key + '}', str(value))
    return text


def parse_object(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.I)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError('Expected a JSON object')
    return obj


def checked_call(client, text, fields, search=False):
    response = client.call(text, search=search)
    try:
        obj = parse_object(response['text'])
        for key, kind in fields.items():
            if key not in obj or type(obj[key]) is not kind:
                raise ValueError(f'Missing/invalid diagnostic field: {key}')
    except (ValueError, TypeError):
        client.forget(response)
        raise
    return obj


def retrieval_test(case, auditor, grader):
    gold, urls = case['gold_urls'], case['retrieved_urls']
    retrieved_ids = {identity(u) for u in urls}
    missing = [u for u in gold if identity(u) not in retrieved_ids]
    exact = [u for u in urls if identity(u) in {identity(g) for g in gold}]
    result = {'gold_urls': gold, 'retrieved_urls': urls, 'search_queries': case['search_queries'],
              'exact_urls': exact, 'equivalent_urls': [], 'query_reasonable': None}
    # Multiple gold pages must ALL be represented, not merely one of them.
    if not missing:
        return {**result, 'match_type': 'exact', 'evidence_retrieved': True}
    evidence = checked_call(auditor, prompt('gold_evidence', {**case, 'gold_urls': missing}),
                            {'key_evidence': str, 'gold_evidence_urls': list}, search=True)
    if not evidence['key_evidence'].strip() or not evidence['gold_evidence_urls']:
        raise ValueError('Gold evidence extraction returned no supporting evidence')
    if any(not isinstance(u, str) or identity(u) not in {identity(g) for g in missing}
           for u in evidence['gold_evidence_urls']):
        raise ValueError('Gold evidence cited an unlisted URL')
    result.update(gold_key_evidence=evidence['key_evidence'], gold_evidence_urls=evidence['gold_evidence_urls'])
    if urls:
        audit = checked_call(auditor, prompt('url_audit', case, gold_key_evidence=evidence['key_evidence']),
            {'evidence_match': bool, 'retrieved_key_evidence': str, 'equivalent_urls': list, 'reason': str}, search=True)
        resolved = []
        for candidate in audit['equivalent_urls']:
            found = [u for u in urls if isinstance(candidate, str) and identity(u) == identity(candidate)]
            if not found:
                raise ValueError('Equivalent URL was not in the retrieved collection')
            resolved.append(found[0])
        if audit['evidence_match'] and (not resolved or not audit['retrieved_key_evidence'].strip()):
            raise ValueError('Positive URL match requires both URL and evidence')
        result.update(equivalent_urls=list(dict.fromkeys(resolved)),
                      retrieved_key_evidence=audit['retrieved_key_evidence'], reason=audit['reason'])
        if audit['evidence_match']:
            return {**result, 'match_type': 'equivalent', 'evidence_retrieved': True}
    queries = checked_call(grader, prompt('queries', case), {'query_reasonable': bool, 'reason': str})
    return {**result, 'match_type': 'none', 'evidence_retrieved': False,
            'query_reasonable': queries['query_reasonable'], 'query_reason': queries['reason']}


def post_retrieval_test(case, test_model, grader):
    gold = checked_call(test_model, prompt('rag', case),
                       {'rag_answer': str, 'page_accessible': bool, 'used_urls': list, 'rag_reasoning': str}, search=True)
    if not gold['page_accessible'] or not gold['rag_answer'].strip():
        raise ValueError('Gold-page access failed; do not attribute an operational failure to reasoning')
    judged = checked_call(grader, prompt('grade', case, candidate_answer=gold['rag_answer']),
                          {'answer_correct': bool, 'reason': str})
    gold = {'answer': gold['rag_answer'], 'used_urls': gold['used_urls'],
            'reasoning': gold['rag_reasoning'], **judged}
    if not gold['answer_correct']:
        return gold, {'ran': False, 'reason': 'Gold-only failed; gold+noise skipped'}
    # Always include real gold URLs, even if a different URL was judged equivalent.
    context = list(dict.fromkeys(case['retrieved_urls'] + case['gold_urls']))
    noise = checked_call(test_model, prompt('gold_noise', case, context_urls=context),
                         {'answer': str, 'used_urls': list, 'reasoning': str}, search=True)
    judged = checked_call(grader, prompt('grade', case, candidate_answer=noise['answer']),
                          {'answer_correct': bool, 'reason': str})
    return gold, {**noise, **judged, 'ran': True, 'context_urls': context}


def decide(retrieval, gold, noise):
    flags = dict.fromkeys(FLAGS, False)
    if not retrieval['evidence_retrieved']:
        flags['retrieval' if retrieval['query_reasonable'] else 'pre_retrieval'] = True
    if not gold['answer_correct']:
        flags['post_retrieval_interpretation'] = True
    elif noise['ran'] and not noise['answer_correct']:
        flags['post_retrieval_distraction'] = True
    flags['randomness'] = not any(flags.values())
    return {'primary_category': 'system-related', 'system_checklist': flags,
            'system_categories': [name for name, value in flags.items() if value]}


def diagnose(case, test_model, auditor, grader, human_category=None):
    if human_category:
        return {**case, 'status': 'ok', 'diagnosis': {'primary_category': 'system-unrelated',
                'human_category': human_category, 'system_checklist': {}, 'system_categories': []}}
    if not case['gold_urls']:
        raise ValueError('No gold URLs supplied')
    # A retrieval failure never prevents the independent post-retrieval probe.
    with ThreadPoolExecutor(max_workers=2) as pool:
        retrieval_job = pool.submit(retrieval_test, case, auditor, grader)
        post_job = pool.submit(post_retrieval_test, case, test_model, grader)
        retrieval = retrieval_job.result()
        gold, noise = post_job.result()
    return {**case, 'status': 'ok', 'diagnosis': decide(retrieval, gold, noise),
            'retrieval': retrieval, 'gold_only': gold, 'gold_plus_noise': noise}
