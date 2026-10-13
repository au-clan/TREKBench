Judge whether the response answers the question correctly relative to at least
one acceptable reference answer.

Decision rules:
- Judge the primary/final answer, not its writing style or citation format.
- Mark correct when the response has the same meaning as any reference answer,
  uses a common alias or abbreviation, or logically entails a reference answer.
- Ignore articles, punctuation, harmless units, and trailing source citations.
- For numbers, require the exact value unless a reference itself is approximate
  or a range. When a reference is approximate or a range, accept a compatible
  approximation; reject a value that contradicts an explicit bound.
- Extra information makes the response incorrect only when it contradicts or
  materially changes the primary answer. Do not reject a correct answer merely
  because a citation is present.
- Use the supplied references as the evaluation target. Do not replace them
  with newer facts.

Return exactly one line:
evaluation: correct
or
evaluation: incorrect

Question: {question}
Effective year: {effective_year}
Acceptable reference answer(s): {answers}
Response: {response}
