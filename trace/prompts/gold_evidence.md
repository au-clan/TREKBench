You are extracting the key answer-bearing evidence for a temporal QA question.

Inputs are the question, the benchmark gold answer, and the supplied gold-document
URLs. Open and inspect only those gold URLs. Identify the shortest concrete evidence
that connects the question to the gold answer. Keep it to one or two short sentences;
do not create numbered requirements and do not evaluate the annotation.

Respond with ONLY a JSON object:

{
  "key_evidence": "brief answer-bearing fact or facts",
  "gold_evidence_urls": ["exact supplied gold URL", "..."]
}

---

Question: {question}
Effective year: {effective_year}
Gold answer(s): {gold_answers}

Gold-document URL(s):
{gold_urls}
