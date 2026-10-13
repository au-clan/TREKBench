You are running a controlled GOLD-PLUS-NOISE probe for a failed web-search
question. The URLs below are the complete allowed context. They may contain
relevant, irrelevant, conflicting, or outdated pages.

Open and reason from these URLs only. Do not use unlisted sources or general web
knowledge. Decide which evidence is relevant and current for the question, and
return the best concrete answer derived from the allowed context.

Resolve relative time expressions against the effective year:
- "last year" means the effective year itself.
- "current", "currently", "this year", "most recent", "now", "latest", and
  "to date" mean as of the end of the effective year.

Respond with ONLY a JSON object:

{
  "answer": "best concrete answer derived from the allowed context",
  "used_urls": ["exact allowed URL used", "..."],
  "reasoning": "one or two concise sentences identifying the decisive evidence"
}

---

Question: {question}
Effective year: {effective_year}

Allowed context URLs:
{context_urls}
