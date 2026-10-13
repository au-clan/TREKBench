"""The four Trek-Ctrl baselines and the original dataset-specific judges."""
import json
import re
import string
from datetime import datetime
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_CHARS = 640_000  # Original experiment's per-document head/tail cap.
CONTEXTS = {"gold": "gold", "OO": "outdated", "NMD": "t1", "SBA": "t2", "SAA": "t2"}
METHODS = ("direct", "recency", "reranking", "filtering", "crag")


@lru_cache(maxsize=None)
def template(name):
    return (HERE.parent / "prompts" / (name + ".md")).read_text(encoding="utf-8")


def documents(task, condition):
    return [task["documents"][i] for i in task["contexts"][CONTEXTS[condition]]]


def render_docs(docs, show_url=False, max_chars=MAX_CHARS):
    blocks = []
    for i, doc in enumerate(docs, 1):
        text = (doc.get("text") or "").strip()
        if max_chars and len(text) > max_chars:
            marker = "\n[... document truncated for context length ...]\n"
            remaining = max_chars - len(marker)
            head = remaining // 2
            text = text[:head] + marker + text[-(remaining - head):]
        lines = [f"[Document {i} — {doc.get('date') or 'date unknown'}]"]
        if show_url and doc.get("url"):
            lines.append(f"Source URL: {doc['url']}")
        blocks.append("\n".join(lines + [text]))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def prompt(name, task, **values):
    return template(name).format(question=task["question"], effective_year=task["effective_year"], **values)


def parse_ids(text, total):
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        payload = json.loads(stripped)
        values = payload.get("keep_document_ids") if isinstance(payload, dict) else None
    except (ValueError, TypeError):
        values = None
    if not isinstance(values, list):
        match = re.search(r'"?keep_document_ids"?\s*:\s*\[([^]]*)\]', stripped, re.I)
        values = re.findall(r"\b(?:Document\s*)?(\d+)\b", match[1], re.I) if match else []
    ids = []
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= total and number not in ids:
            ids.append(number)
    return ids


def fallback_rank(docs, task):
    # Preserve the original selector fallback, including retaining ALL documents.
    from dateutil import parser
    terms = {s.lower() for s in re.findall(r"[A-Za-z0-9]+", task["question"]) if len(s) > 2}
    try:
        cutoff = datetime(int(task["effective_year"]), 12, 31)
    except (ValueError, TypeError):
        cutoff = None

    def score(pair):
        index, doc = pair
        text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
        relevance = sum(term in text for term in terms) / max(1, len(terms))
        timestamp = float("-inf")
        if doc.get("date"):
            try:
                date = parser.parse(str(doc["date"])).replace(tzinfo=None)
                timestamp = date.timestamp()
                if cutoff and date > cutoff:
                    timestamp -= 100 * 365 * 24 * 3600
            except (ValueError, TypeError, OverflowError):
                pass
        return relevance, timestamp, -index
    return [i for i, _ in sorted(enumerate(docs, 1), key=score, reverse=True)]


def answer(client, task, method, condition, reranker_config=None):
    docs = documents(task, condition)
    show_url = condition == "SAA"
    cap = task["max_chars_per_doc"][CONTEXTS[condition]]
    ids, calls, trace = list(range(1, len(docs) + 1)), [], {}
    if method == "reranking":
        from .rerank import rank
        ranking, scores = rank(docs, task, cap, reranker_config or {})
        ids = ranking[:10]
        trace.update(ranked_document_ids=ranking, document_scores=scores)
    elif method == 'recency':
        from .rerank import date_value
        ids = sorted(ids, key=lambda i: date_value(docs[i-1].get('date')) or datetime.min.date(), reverse=True)[:1]
    elif method in ("filtering", "crag"):
        name = "longseal_context_filter" if method == "filtering" else "longseal_crag_extract"
        stage = client.call(prompt(name, task, docs_block=render_docs(docs, show_url, cap)))
        calls.append({"stage": "filter" if method == "filtering" else "extract", **stage})
        if method == "filtering":
            ids = parse_ids(stage["text"], len(docs))
            trace["filter_fallback"] = not bool(ids)
            if not ids:
                ids = fallback_rank(docs, task)
        else:
            final = client.call(prompt("longseal_crag_resolve", task, extracted_evidence=stage["text"]))
            calls.append({"stage": "resolve", **final})
            return final["text"], {"calls": calls, "selected_document_ids": ids}
    final = client.call(prompt("longseal", task,
                               docs_block=render_docs([docs[i - 1] for i in ids], show_url, cap)))
    calls.append({"stage": "answer", **final})
    trace.update(calls=calls, selected_document_ids=ids)
    if method == "reranking":
        gold = set(task["gold_document_ids"][CONTEXTS[condition]])
        trace["gold_document_recall_at_k"] = {
            str(k): len(gold.intersection(ranking[:k])) / len(gold) if gold else None
            for k in (1, 3, 5, 10)}
    return final["text"], trace


def reference_match(prediction, references):
    citation = r"\s*(?:\(\s*)?\[[^\]]+\]\(https?://[^)]+\)(?:\s*\))?\s*$"
    while True:
        stripped = re.sub(citation, "", prediction, flags=re.I)
        if stripped == prediction:
            break
        prediction = stripped

    def normalize(text):
        text = re.sub(r"[*_`#]+", "", text or "").lower()
        return re.sub(r"\s+", " ", text).strip(string.punctuation + "。，、！？；：“”‘’" + string.whitespace)
    value = normalize(prediction)
    return any(normalize(ref) and value == normalize(ref) for ref in references)


def judge_prompt(dataset, task, prediction):
    name = dataset + "_judge"
    if dataset == "infodeepseek" and task["false_premise"] is True:
        name = "infodeepseek_false_premise_judge"
    target = " | ".join(task["answers"])
    return template(name).format(question=task["question"], query=task["question"],
        answers=target, target=target, reference_answer=target, response=prediction,
        predicted_answer=prediction, candidate_answer=prediction, effective_year=task["effective_year"])


def verdict(dataset, text):
    # Invalid outputs are retryable errors, not incorrect answers.
    text = text.strip().rstrip(".").strip()
    if dataset == "freshqa":
        match = re.fullmatch(r"evaluation\s*:\s*(correct|incorrect)", text, re.I)
        result = match[1].lower() if match else None
    elif dataset == "infodeepseek":
        result = {"yes": "correct", "no": "incorrect"}.get(text.lower())
    else:
        result = {"A": "correct", "B": "incorrect", "C": "not_attempted"}.get(text.upper())
    if result is None:
        raise ValueError(f"Invalid {dataset} judge verdict: {text[:100]!r}")
    return result


def grade(client, dataset, task, prediction):
    if dataset == "freshqa" and reference_match(prediction, task["answers"]):
        return {"correct": True, "verdict": "correct", "method": "reference_match"}
    response = client.call(judge_prompt(dataset, task, prediction))
    try:
        result = verdict(dataset, response["text"])
    except ValueError:
        client.forget(response)
        raise
    return {"correct": result == "correct", "verdict": result, "method": "llm", "call": response}
