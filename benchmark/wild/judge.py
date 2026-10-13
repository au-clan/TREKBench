"""Grade candidate answers AGAINST supplied Gold; this does not verify Gold truth."""
from __future__ import annotations

import argparse
import json
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import (DATASETS, HERE, append_jsonl, ensure_run, fingerprint, key,
                    latest_records, load_configs, load_tasks, make_client,
                    safe_error, save_json)
from benchmark.data import DATA

TRAILING_CITATION = re.compile(r"\s*(?:\(\s*)?\[[^\]]+\]\(https?://[^)]+\)(?:\s*\))?\s*$", re.I)


def deterministic_reference_match(prediction, references):
    # Same conservative equality check as the original FreshQA evaluator.
    while True:
        stripped = TRAILING_CITATION.sub("", prediction)
        if stripped == prediction:
            break
        prediction = stripped
    def normalize(value):
        value = re.sub(r"[*_`#]+", "", value or "").lower()
        return re.sub(r"\s+", " ", value).strip(string.punctuation + "。，、！？；：“”‘’" + string.whitespace)
    normalized = normalize(prediction)
    return any(normalize(ref) and normalized == normalize(ref) for ref in references)


def build_judge_prompt(dataset, task, prediction, configs=None):
    config = (configs or load_configs())[dataset]["evaluator"]
    metadata = task.get("metadata") or {}
    template = config["template"]
    if metadata.get("false_premise") is True and config.get("false_premise_template"):
        template = config["false_premise_template"]
    answers = task.get("answers") or [task["answer"]]
    target = " | ".join(answers)
    prompt = (HERE.parent / template).read_text(encoding="utf-8").format(
        question=task["question"], query=task["question"], answers=target,
        response=prediction, target=target, predicted_answer=prediction,
        reference_answer=target, candidate_answer=prediction,
        effective_year=metadata.get("effective_year", ""),
    )
    return prompt, template


def parse_verdict(dataset, text):
    # Invalid judge output is a failed request, never silently counted incorrect.
    text = text.strip().rstrip(".").strip()
    if dataset == "freshqa":
        match = re.fullmatch(r"evaluation\s*:\s*(correct|incorrect)", text, re.I)
        verdict = match.group(1).lower() if match else None
    elif dataset == "infodeepseek":
        verdict = {"yes": "correct", "no": "incorrect"}.get(text.lower())
    else:
        verdict = {"A": "correct", "B": "incorrect", "C": "not_attempted"}.get(text.upper())
    if verdict is None:
        raise ValueError(f"Invalid {dataset} judge verdict: {text[:160]!r}")
    return verdict


def grade_one(client, dataset, task, prediction, model="gpt-5.4-mini", effort="low"):
    prompt, template = build_judge_prompt(dataset, task, prediction)
    row = {"dataset": dataset, "task_id": task["id"], "question": task["question"],
           "prediction": prediction, "answers": task.get("answers") or [task["answer"]],
           "judge_model": model, "judge_template": template, "judge_prompt": prompt}
    try:
        if dataset == "freshqa" and deterministic_reference_match(prediction, row["answers"]):
            verdict, text, usage, method = "correct", "evaluation: correct", None, "deterministic_reference_match"
        else:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], reasoning_effort=effort)
            if not response.choices or response.choices[0].finish_reason != "stop":
                raise ValueError("Judge did not finish normally")
            text = response.choices[0].message.content or ""
            row["judge_text"] = text
            verdict = parse_verdict(dataset, text)
            usage = response.usage.model_dump(mode="json") if response.usage else None
            method = "llm"
        row.update(status="ok", verdict=verdict, correct=verdict == "correct",
                   judge_text=text, judge_method=method, usage=usage)
    except Exception as exc:
        row.update(status="error", correct=None, error_type=type(exc).__name__, error=safe_error(exc))
    return row


def summarize(predictions, grades):
    result = {}
    for dataset in (*DATASETS, "all"):
        ids = [identity for identity in predictions if dataset == "all" or identity[0] == dataset]
        eligible = [identity for identity in ids if predictions[identity].get("status", "ok") == "ok"]
        complete = [grades[identity] for identity in eligible if grades.get(identity, {}).get("status") == "ok"]
        correct = sum(row["correct"] is True for row in complete)
        result[dataset] = {
            "total": len(ids), "generation_failed_or_missing": len(ids) - len(eligible),
            "judged": len(complete), "judge_failed_or_pending": len(eligible) - len(complete),
            "correct": correct, "not_attempted": sum(r["verdict"] == "not_attempted" for r in complete),
            "accuracy": correct / len(ids) if ids and len(complete) == len(ids) else None,
            "accuracy_on_judged": correct / len(complete) if complete else None,
        }
    active = [result[ds]['accuracy'] for ds in DATASETS if result[ds]['total']]
    result['all']['micro_accuracy'] = result['all']['accuracy']
    result['all']['macro_accuracy'] = (sum(active) / len(active)
                                      if active and all(v is not None for v in active) else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning", default="low", choices=["none", "low", "medium", "high"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be >= 1")
    if not args.predictions.is_file():
        parser.error("Predictions file does not exist")
    predictions, tasks = latest_records(args.predictions), load_tasks(args.data_dir)
    generation_manifest = args.predictions.parent / "manifest.json"
    if generation_manifest.exists():
        manifest = json.loads(generation_manifest.read_text())
        if manifest.get("stage") == "generation":
            for task in manifest["tasks"]:
                predictions.setdefault(key(task), {**task, "status": "missing"})
    if not predictions or any(identity not in tasks for identity in predictions):
        parser.error("Predictions must be nonempty and use known dataset/task IDs")
    jobs = []
    for (dataset, task_id), prediction in predictions.items():
        if prediction.get("status", "ok") == "ok":
            if not isinstance(prediction.get("prediction"), str):
                parser.error(f"Missing text prediction: {task_id}")
            task = tasks[(dataset, task_id)]
            prompt, template = build_judge_prompt(dataset, task, prediction["prediction"])
            jobs.append((dataset, task, prediction["prediction"], prompt, template))
    ensure_run(args.output, {"stage": "judge", "dry_run": args.dry_run,
                            "model": args.model, "reasoning": args.reasoning,
                            "input_sha256": fingerprint(jobs), "prediction_ids": sorted(predictions)})
    if args.dry_run:
        save_json(args.output / "judge_requests.json", [{"dataset": ds, "task_id": task["id"],
            "template": template, "prompt": prompt} for ds, task, _, prompt, template in jobs])
        print(f"Prepared {len(jobs)} judge prompts; no API calls.")
        return 0
    destination = args.output / "grades.jsonl"
    grades = latest_records(destination)
    pending = [job for job in jobs if grades.get((job[0], job[1]["id"]), {}).get("status") != "ok"]
    if pending:
        needs_llm = any(ds != "freshqa" or not deterministic_reference_match(pred, task.get("answers") or [task["answer"]])
                        for ds, task, pred, _, _ in pending)
        client = make_client() if needs_llm else None
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(grade_one, client, ds, task, pred, args.model, args.reasoning)
                       for ds, task, pred, _, _ in pending]
            for future in as_completed(futures):
                row = future.result()
                append_jsonl(destination, row)
                grades[key(row)] = row
                print(row["dataset"], row["task_id"], row.get("verdict", "error"), flush=True)
    summary = summarize(predictions, grades)
    save_json(args.output / "scores.json", summary)
    print(json.dumps(summary, indent=2))
    return int(summary["all"]["judged"] != summary["all"]["total"])


if __name__ == "__main__":
    raise SystemExit(main())
