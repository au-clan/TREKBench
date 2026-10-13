"""Generate answers only. Gold answers, URLs and reference queries are never sent."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import (DATASETS, HERE, answer_prompt, append_jsonl, ensure_run,
                    fingerprint, latest_records, load_configs, load_tasks,
                    make_client, safe_error, save_json)
from benchmark.data import DATA


def request_kwargs(model, effort, search, prompt):
    args = {"model": model, "input": prompt, "reasoning": {"effort": effort}}
    if search:
        args.update(tools=[{"type": "web_search"}], include=["web_search_call.action.sources"])
    return args


def extract_trace(raw):
    calls = [item for item in raw.get("output", []) if item.get("type") == "web_search_call"]
    urls, citations = [], []
    for item in calls:
        action = item.get("action") or {}
        if str(action.get("url", "")).startswith(("http://", "https://")):
            urls.append(action["url"])
        urls.extend(s["url"] for s in action.get("sources") or [] if s.get("url"))
    for item in raw.get("output", []):
        for content in item.get("content") or []:
            citations.extend(a["url"] for a in content.get("annotations") or []
                             if a.get("type") == "url_citation" and a.get("url"))
    return {"search_calls": calls, "retrieved_urls": list(dict.fromkeys(urls)),
            "citation_urls": list(dict.fromkeys(citations))}


def generate_one(client, dataset, task, request):
    row = {"dataset": dataset, "task_id": task["id"], "question": task["question"],
           "model": request["model"], "reasoning": request["reasoning"]["effort"],
           "search_enabled": bool(request.get("tools")), "prompt": request["input"]}
    try:
        result = client.responses.create(**request)
        raw = result.model_dump(mode="json")
        if raw.get("status") != "completed" or not result.output_text.strip():
            raise ValueError(f"Non-complete/empty response: {raw.get('status')}, {raw.get('incomplete_details')}")
        row.update(status="ok", prediction=result.output_text, response_id=raw.get("id"),
                   usage=raw.get("usage"), **extract_trace(raw))
    except Exception as exc:
        row.update(status="error", error_type=type(exc).__name__, error=safe_error(exc))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--limit", type=int, default=1, help="First N per dataset; 0 = all.")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning", default="high", choices=["none", "low", "medium", "high"])
    parser.add_argument("--search", action="store_true", help="Expose native web search (not forced).")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Write requests only; no API calls or scores.")
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1 or len(args.datasets) != len(set(args.datasets)):
        parser.error("limit >= 0, workers >= 1, and distinct datasets required")
    tasks, configs = load_tasks(args.data_dir), load_configs()
    jobs = []
    for dataset in args.datasets:
        selected = [task for (name, _), task in tasks.items() if name == dataset]
        for task in selected[:args.limit or None]:
            prompt = answer_prompt(dataset, task, configs)
            jobs.append((dataset, task, request_kwargs(args.model, args.reasoning, args.search, prompt)))
    manifest = {"stage": "generation", "dry_run": args.dry_run,
                "model": args.model, "reasoning": args.reasoning, "search": args.search,
                "input_sha256": fingerprint(jobs),
                "tasks": [{"dataset": ds, "task_id": task["id"]} for ds, task, _ in jobs]}
    ensure_run(args.output, manifest)
    if args.dry_run:
        save_json(args.output / "requests.json", [
            {"dataset": ds, "task_id": task["id"], "request": req} for ds, task, req in jobs])
        print(f"Prepared {len(jobs)} requests; no API calls. {args.output / 'requests.json'}")
        return 0
    destination = args.output / "predictions.jsonl"
    existing = latest_records(destination)
    pending = [job for job in jobs if existing.get((job[0], job[1]["id"]), {}).get("status") != "ok"]
    if pending:
        client = make_client()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(generate_one, client, *job) for job in pending]
            for future in as_completed(futures):
                row = future.result()
                append_jsonl(destination, row)
                existing[(row["dataset"], row["task_id"])] = row
                print(row["dataset"], row["task_id"], row["status"], flush=True)
    ok = sum(existing.get((ds, task["id"]), {}).get("status") == "ok" for ds, task, _ in jobs)
    save_json(args.output / "generation_summary.json", {"requested": len(jobs), "completed": ok, "failed": len(jobs) - ok})
    print(f"Completed {ok}/{len(jobs)}. Rerun the same command to retry failures only.")
    return int(ok != len(jobs))


if __name__ == "__main__":
    raise SystemExit(main())
