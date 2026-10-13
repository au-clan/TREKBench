Assess whether the issued search queries could reasonably retrieve the evidence
needed for this question. Consider the entity, requested fact, temporal scope,
and any required multi-hop decomposition. Judge the queries, not the final answer.
Return only JSON: {"query_reasonable": true | false, "reason": "brief explanation"}.

Question: {question}
Effective year: {effective_year}
Gold answer(s): {gold_answers}
Gold URLs:
{gold_urls}
Issued queries:
{search_queries}
