from agent.agent import answer_with_grounding


def _evidence():
    return [{
        "text": "Ravi Kumar was arrested on 10 March 2026.",
        "document": "FIR.pdf",
        "page": 2,
        "chunk_id": "fir-2-1",
    }]


def test_conversational_candidate_is_still_verified():
    result = answer_with_grounding(
        "When was Ravi Kumar arrested?",
        _evidence(),
        candidate_generator=lambda q, c: (
            "Ravi Kumar was arrested on 10 March 2026."
        ),
        conversation_history=[
            {"role": "user", "content": "Who is the accused?"},
            {"role": "assistant", "content": "Ravi Kumar."},
        ],
    )
    assert result["claims"]
    assert result["claims"][0]["status"] == "SUPPORTED"
    assert result["citations"]
    assert result["confidence"] == "HIGH"


def test_unsupported_llm_claim_is_rejected():
    result = answer_with_grounding(
        "What happened next?",
        _evidence(),
        candidate_generator=lambda q, c: (
            "Ravi Kumar was convicted by the Supreme Court on 20 March 2026."
        ),
    )
    assert result["answer"] == "Not verified from the provided sources."
    assert result["unsupported_claims"]
    assert result["unsupported_claims"][0]["status"] != "SUPPORTED"
