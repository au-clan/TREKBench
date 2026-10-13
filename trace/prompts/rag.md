You are running a controlled GOLD-ONLY probe for a failed web-search question.
Open and reason from only the supplied ground-truth page URL(s). Treat these
URLs as the authoritative context intended to answer the question. Do not use
unlisted sources or general web knowledge.

If a supplied URL is blocked or cannot be opened directly, recover an
accessible representation of that same page (for example a cached copy,
search-result rendering, mobile/alternate form, or mirror) and continue to
answer from the recovered page content.

Answer the question directly. Inspect every relevant supplied page and perform
any necessary entity selection, multi-hop reasoning, filtering, counting, date
calculation, or comparison. Return the best concrete answer derived from the
supplied URLs.

`page_accessible` reports whether the content of at least one supplied
ground-truth page was available directly or through such a representation.
`used_urls` must contain the original supplied URL(s) whose content contributed
to the answer.

Resolve relative time expressions against the effective year:
- "last year" means the effective year itself.
- "current", "currently", "this year", "most recent", "now", "latest", and
  "to date" mean as of the end of the effective year.

Respond with ONLY a JSON object:

{
  "rag_answer": "concrete best answer derived from the supplied pages",
  "page_accessible": true | false,
  "used_urls": ["exact supplied URL used", "..."],
  "rag_reasoning": "one or two concise sentences identifying the supporting fact"
}

---

Question: {question}
Effective year: {effective_year}

Ground-truth page URL(s):
{gold_urls}
