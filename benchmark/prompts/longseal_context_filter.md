Filter the documents before another model answers the question as of
{effective_year}. Keep documents containing information that directly satisfies
the complete information need. Remove documents that are irrelevant, fail any
requirement in the question, or contain untrustworthy information. If multiple
documents contain information that satisfies the same need, retain the latest
information that is valid for the requested time and remove superseded versions.
There is no required or maximum number of documents to keep. Do not answer the
question.

Return only JSON with all one-based document IDs that should be retained:
{{"keep_document_ids": [2, 5, 9]}}

Question: {question}

{docs_block}
