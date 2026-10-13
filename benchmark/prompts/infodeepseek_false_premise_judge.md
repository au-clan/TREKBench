Given a question and its ground-truth answer, determine
whether the candidate answer correctly answers the given
question. Pay attention to the following points:
1. This question has a false premise, which has been pointed
out in the groundtruth answer. If the candidate answer does
not point out or correct this false premise, it is incorrect.
2. If the false premise pointed out by the candidate answer
is different from the groundtruth answer (time, place, event,
person, meaning inconsistent), it is incorrect.
3. If the groundtruth answer still answers the question in
addition to pointing out the false premise, the candidate
answer should also answer the question. In this case, if the
candidate answer does not answer the question or the meaning
of the answer content is inconsistent with the groundtruth
answer (pay attention to the consistency of time, place,
person, and quantity), it is incorrect.

Just answer Yes or No.

Given question: {query}
Groundtruth answer: {reference_answer}
Candidate answer: {candidate_answer}
Does the candidate answer correctly answer the given question?
