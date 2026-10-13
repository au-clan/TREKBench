# TREKBench

Release of **TREKBench-Wild**, **TREKBench-Ctrl**, and **TemporalTrace**.


```text
benchmark/           Wild/Ctrl runners, shared prompts and data loader
  wild/              Open-web answering and dataset-specific judging
  ctrl/              Frozen-context baselines and live reranking
data/                One dataset containing both Wild and Ctrl views
trace/               Counterfactual failure attribution and its prompts
```

## Installation

Python 3.10+; run commands from this directory.

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY"
```

Use environment variables for keys. Never commit keys or generated outputs.
API calls incur costs. Model access, context limits, and supported reasoning
settings depend on the provider. The package does not substitute another model
or silently shorten the context if a request fails.

## Data

`data/trekbench.jsonl.gz` contains **144 records**: FreshQA 49,
InfoDeepSeek 31, and SealQA 64. Each record has `dataset`, `id`, `wild`, and `ctrl`.
The same question is joined across the two views. `id` is the original Wild ID;
`ctrl.id` additionally preserves its original controlled-context ID.

```python
from benchmark.data import load_records
row = load_records()[0]
print(row["wild"]["question"], row["ctrl"]["contexts"].keys())
```

Wild stores the question, references, gold URLs, and necessary metadata. Ctrl
adds a deduplicated document pool, context indices (zero-based), gold-document
IDs within each context (one-based), and the original per-document length caps.
Gold IDs and references are never passed to the answer model or reranker.
SAA/SBA use the same document indices and order.

| Condition | Distractors (gold is additional) |
|---|---|
| `gold` | None; Gold Only reference |
| `OO` | Up to 20 outdated documents |
| `NMD` | Up to 10 outdated + 10 T1 documents |
| `SAA` | Sampled NMD background + T2A/T2B, with source URL headers |
| `SBA` | Same as SAA, without source URL headers |

Shortages are retained; counts are not always 20. The two source conditions
differ only in explicit URL headers, not their bodies. Source names can still
occur inside document text. The frozen per-document truncation rule is retained.
No new total-context cap is imposed.

Selection metadata and input hashes are in `data/manifest.json`. These are the
original selected cohorts, **not** the full upstream datasets. Known outdated
or erroneous references and false-premise questions have not been removed or
corrected. Wild/ Ctrl reference-year fields are kept separately: the Ctrl
construction's 2025 fallback for InfoDeepSeek is not injected into Wild prompts.

## Wild

```bash
# No API call: prepare one request per dataset
python -m benchmark.wild.run --search --limit 1 --dry-run --output runs/wild_preview

# Generate, then independently judge. --limit 0 runs all 144 questions.
python -m benchmark.wild.run --model gpt-5.4 --reasoning high --search \
  --limit 1 --workers 3 --output runs/wild
python -m benchmark.wild.judge --predictions runs/wild/predictions.jsonl \
  --output runs/wild_judged
```

Omit `--search` for the closed-book condition. This minimal search runner uses
OpenAI Responses native web search; it is not an implementation of every
provider/search backend in the full experimental repository. It records search
actions and exposed retrieved URLs separately from final citations. See the
[official API documentation](https://developers.openai.com/api/docs/guides/tools-web-search).
Provider-exposed sources are not a guarantee of access to every internal search
result. A search-enabled request may still make no search calls.

Scoring preserves the source protocols: FreshQA reference matching followed by
its semantic judge; InfoDeepSeek normal/false-premise routing; SealQA A/B/C.
Invalid verdicts are errors, not wrong answers. Output includes dataset accuracy,
micro accuracy over examples, and macro accuracy over the evaluated datasets.
Incomplete runs do not receive a completed accuracy score.

## Ctrl

```bash
python -m benchmark.ctrl.run --datasets all --methods direct \
  --conditions gold OO NMD SAA SBA --limit 1 --dry-run

python -m benchmark.ctrl.run --datasets all --methods direct \
  --conditions gold OO NMD SAA SBA --limit 1 --judge --out runs/ctrl
```

Available methods: `direct`, `recency` (latest one document), `reranking`,
`filtering`, and `crag`. `--methods all` enables them all. Gold Only is generated
once per question using Direct, not once per method. Use `--limit 0` for all data.

`benchmark/ctrl/config.json` sets the answer/judge model, endpoint, key environment
variable, and native parameters. Chat-completions-compatible providers can be
used for Ctrl; for example, change `base_url`, `api_key_env`, `model`, and
`parameters` to the values required by your provider. No search tool is used.

Reranking calls `qwen/qwen3-reranker-8b` through OpenRouter and requires
`OPENROUTER_API_KEY`. It chunks documents into 256-word windows with 32-word
overlap, takes the maximum semantic score per document, multiplies by Gaussian
temporal relevance, and keeps the top 10 documents. Dates after the cutoff score
zero; missing dates use the mean available temporal score (one if none exist).
Gold-document Recall@k is computed only after ranking. No previous rankings or
scores are distributed. Optional `reranker` config keys are `endpoint`, `model`,
and `api_key_env`.

Filtering has no fixed document-count limit. C-RAG uses evidence extraction
followed by conflict resolution. Both retain the existing prompts. The existing
filter fallback retains all documents if no valid IDs are selected.

## Trace

```bash
python -m trace.run --predictions runs/wild/predictions.jsonl \
  --grades runs/wild_judged/grades.jsonl --dry-run

python -m trace.run --predictions runs/wild/predictions.jsonl \
  --grades runs/wild_judged/grades.jsonl --workers 2 --out runs/trace
```
Trace reads only successfully judged failures. Both counterfactuals use the
original prediction's model and reasoning effort, not the judge model. The
minimal URL-based probes require OpenAI Responses models supporting web access.
The two diagnostic branches are independent:

1. **Retrieval:** all gold URLs present, or matching key evidence in the retrieved
   URL collection. Only when neither match succeeds are query adequacy and
   pre-retrieval/retrieval attribution considered.
2. **Post-retrieval:** incorrect gold-only implies interpretation. Otherwise run
   gold plus the original retrieved URLs, inserting every missing real gold URL.
   Failure then implies distraction. Multiple stage labels are allowed.
