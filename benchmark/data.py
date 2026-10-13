"""One dataset, with separate Wild and Ctrl views of each question."""
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DATASETS = ('freshqa', 'infodeepseek', 'sealqa')


@lru_cache(maxsize=4)
def load_records(data_dir=DATA):
    directory = Path(data_dir)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    path = directory / 'trekbench.jsonl.gz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Dataset checksum mismatch')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != manifest['questions'] or len({(r['dataset'], r['id']) for r in rows}) != len(rows):
        raise ValueError('Wrong question count or duplicate IDs')
    return rows
