"""Small shared helpers; no dependency on the parent TemporalLens project."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from benchmark.data import DATA, load_records

HERE = Path(__file__).resolve().parent
DATASETS = ("freshqa", "infodeepseek", "sealqa")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def key(row):
    return row["dataset"], row["task_id"]


def latest_records(path):
    return {key(row): row for row in read_jsonl(path)} if Path(path).exists() else {}


def load_tasks(data_dir=DATA):
    return {(row['dataset'], row['id']): row['wild'] for row in load_records(data_dir)}


def load_configs():
    return json.loads((HERE / "benchmark_configs.json").read_text(encoding="utf-8"))


def effective_year_block(year):
    # Byte-for-byte equivalent to the original benchmark adapter for these inputs.
    if not year or not str(year).isdigit():
        return ""
    return (
        f"Answer as of {year}: the answer must reflect the state of the "
        f"world in {year}. Interpret every relative time expression in the "
        f'question — "current", "currently", "now", "today", "this year", '
        f'"last year", "recent", "latest", "most recent" — as referring to '
        f"{year} (use the end of {year} when a precise point is needed).\n\n"
    )


def answer_prompt(dataset, task, configs=None):
    config = (configs or load_configs())[dataset]
    year = (task.get("metadata") or {}).get("effective_year")
    return config["prompt_template"].format(
        question=task["question"], effective_year_block=effective_year_block(year)
    )


def ensure_run(output, manifest):
    manifest = json.loads(json.dumps(manifest))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Output directory belongs to different inputs/settings. Use a new --output.")
    elif any(output.iterdir()):
        raise ValueError("Output directory is nonempty but has no manifest. Use a new --output.")
    else:
        save_json(path, manifest)


def make_client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY, or use --dry-run (no API key required).")
    from openai import OpenAI
    return OpenAI(timeout=300.0, max_retries=3)


def safe_error(exc):
    text = str(exc)
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text[:500]
