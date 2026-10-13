"""Live Qwen semantic ranking x Gaussian temporal decay; no stored scores."""
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from .api import fingerprint, save_json


def date_value(value):
    raw = str(value or '').strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace('Z', '+00:00')).date()
    except ValueError:
        pass
    for fmt in ('%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y'):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    return date(int(raw), 12, 31) if re.fullmatch(r'\d{4}', raw) else None


def temporal_scores(docs, year):
    match = re.search(r'\b(\d{4})\b', str(year))
    if not match:
        return [1.0] * len(docs)
    cutoff = date(int(match[1]), 12, 31)
    scores = []
    for doc in docs:
        published = date_value(doc.get('date'))
        delta = (cutoff - published).days if published else None
        scores.append(None if delta is None else 0.0 if delta < 0 else
                      math.exp(math.log(0.5) * (delta / 365) ** 2))
    known = [s for s in scores if s is not None]
    mean = sum(known) / len(known) if known else 1.0
    return [mean if s is None else s for s in scores]


def request_scores(query, texts, config):
    endpoint = config.get('endpoint', 'https://openrouter.ai/api/v1/rerank')
    payload = {'model': config.get('model', 'qwen/qwen3-reranker-8b'),
               'query': query, 'documents': texts, 'top_n': len(texts)}
    key = fingerprint({'endpoint': endpoint, 'payload': payload})
    path = Path(config.get('cache', '.cache/rerank')) / (key + '.json')
    if path.exists():
        return json.loads(path.read_text())['scores']
    name = config.get('api_key_env', 'OPENROUTER_API_KEY')
    token = os.environ.get(name)
    if not token:
        raise ValueError(f'Set {name} to run live reranking')
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                body = json.load(response)
            scores = {int(r['index']): float(r['relevance_score']) for r in body.get('results', [])}
            if set(scores) != set(range(len(texts))) or not all(math.isfinite(v) for v in scores.values()):
                raise ValueError('Reranker returned incomplete or invalid scores')
            ordered = [scores[i] for i in range(len(texts))]
            save_json(path, {'scores': ordered})
            return ordered
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code not in (408, 429, 500, 502, 503, 504) or attempt == 4:
                raise RuntimeError(f'Rerank HTTP {code}') from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise RuntimeError('Reranker connection failed') from None
        time.sleep(min(16, 2 ** attempt))


def rank(docs, task, cap, config):
    chunks = []
    for i, doc in enumerate(docs, 1):
        text = str(doc.get('text') or '')
        if cap and len(text) > cap:
            marker = '\n[... document truncated for context length ...]\n'
            remaining = cap - len(marker)
            head = remaining // 2
            text = text[:head] + marker + text[-(remaining-head):]
        words = list(re.finditer(r'\S+', text))
        for start in range(0, max(1, len(words)), 224):
            end = min(start + 256, len(words))
            piece = text[words[start].start():words[end-1].end()] if words else ''
            piece = re.sub(r'\S{12001,}', lambda m: m[0][:6000] + '[...encoded-content-omitted...]' + m[0][-6000:], piece)
            if doc.get('title'):
                piece = f"Title: {doc['title']}\n{piece}"
            chunks.append((i, piece))
            if end == len(words):
                break
    query = f"Question: {task['question']}\nAnswer time frame: {task['effective_year']}"
    semantic = {i: float('-inf') for i in range(1, len(docs)+1)}
    offset = 0
    while offset < len(chunks):
        batch, chars = [], 0
        while offset + len(batch) < len(chunks) and len(batch) < 64:
            entry = chunks[offset + len(batch)]
            if batch and chars + len(entry[1]) > 48000:
                break
            batch.append(entry)
            chars += len(entry[1])
        scores = request_scores(query, [text for _, text in batch], config)
        for (i, _), score in zip(batch, scores):
            semantic[i] = max(semantic[i], score)
        offset += len(batch)
    temporal = temporal_scores(docs, task['effective_year'])
    scores = [{'document_id': i, 'semantic_score': semantic[i], 'temporal_score': temporal[i-1],
               'score': semantic[i] * temporal[i-1]} for i in semantic]
    ordered = sorted(scores, key=lambda s: (-s['score'], s['document_id']))
    return [s['document_id'] for s in ordered], ordered
