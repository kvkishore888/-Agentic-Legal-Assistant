from evaluation.metrics import (
    answer_usefulness, citation_correctness, fabricated_claim_rate,
    groundedness, retrieval_recall_at_k,
)


EVIDENCE = [{"document":"FIR.pdf","page":2,"chunk_id":"FIR_p2_c4",
             "text":"The accused was arrested on 10 March 2026."}]
EXPECTED = [{"document":"FIR.pdf","page":2,"chunk_id":"FIR_p2_c4"}]


def test_retrieval_recall():
    assert retrieval_recall_at_k(EVIDENCE, EXPECTED, 5) == 1.0


def test_citation_correctness():
    assert citation_correctness(EXPECTED, EXPECTED) == 1.0


def test_groundedness_and_fabrication():
    claims = [{"claim":"The accused was arrested on 10 March 2026."}]
    assert groundedness(claims, EVIDENCE) == 1.0
    assert fabricated_claim_rate(claims, EVIDENCE) == 0.0


def test_usefulness():
    assert answer_usefulness("The accused was arrested on 10 March 2026.",
                             "The accused was arrested on 10 March 2026.") == 1.0
