Given a question and its groundtrue answer, determine whether
the candidate answer correctly answers the given question.
Pay attention to the following points:
1. The candidate answer must contain content that is
consistent with the groundtrue answer to be considered correct
(pay attention to the consistency of time, place, person, and
quantity), but more details can be provided.
2. If there are multiple contents/events/persons in the
groundtrue answer, the candidate answer must also contain
all the contents/events/persons to be considered correct.
3. The given question does not have a wrong premise, and
the relevant person/event must exist and be unique. If the
candidate answer proposes a wrong premise or cannot determine
whether the person/event exists, it is a wrong answer.
4. The given question has a clear answer and the given
groundtrue answer must be correct. If the candidate answer
does not answer the question correctly but proposes the need
to further query relevant information, it is a wrong answer.

Just answer Yes or No.

Given Question: {query}
Groundtrue Answer: {reference_answer}
Candidate Answer: {candidate_answer}
Does the candidate answer correctly answer the given question?
