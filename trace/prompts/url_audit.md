You are matching answer-bearing evidence between a gold document and a retrieved
web-document collection.

The short gold key evidence below was extracted in a separate pass from the
question, gold answer, and gold URLs. Use web access only to inspect URLs in the
supplied retrieved collection. Do not use outside URLs or general knowledge.

Look for the same key fact or facts in the retrieved collection. Multiple supplied
pages may jointly provide the evidence. `evidence_match` is true only when the
retrieved pages contain the gold key evidence, not merely when they mention the
same topic. Do not answer the original question. Return the concise matching
evidence and all matching retrieved URLs copied exactly from the input.

Respond with ONLY a JSON object:

{
  "evidence_match": true | false,
  "retrieved_key_evidence": "brief matching fact or facts, or empty",
  "equivalent_urls": ["exact supplied retrieved URL", "..."],
  "reason": "brief explanation"
}

---

Question: {question}
Effective year: {effective_year}
Gold answer(s): {gold_answers}
Gold key evidence: {gold_key_evidence}

Retrieved URL collection:
{retrieved_urls}
