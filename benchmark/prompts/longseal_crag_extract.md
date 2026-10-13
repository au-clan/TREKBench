For every document, extract the passage most useful for answering the
question and identify the answer it supports. Briefly state whether it
directly answers the question, provides competing evidence, or is irrelevant.
Do not answer the question.

Return only a compact JSON array with these fields: document_id,
key_evidence, candidate_answer, relation, and reason.

Question: {question}
Time frame: {effective_year}

{docs_block}
