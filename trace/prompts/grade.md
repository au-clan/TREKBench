Judge whether the candidate answer matches the gold answer for the question.
Allow trivial formatting, units, and paraphrase differences, but be strict about
entities, dates, counts, and other substantive facts.

Respond with ONLY a JSON object:

{
  "answer_correct": true | false,
  "reason": "one concise, human-checkable sentence"
}

Question: {question}
Gold answer(s): {gold_answers}
Candidate answer: {candidate_answer}
