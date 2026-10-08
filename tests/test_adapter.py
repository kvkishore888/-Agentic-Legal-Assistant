from app.backend_adapter import normalize_answer


def test_normalize_answer_preserves_verifiability_fields():
    result = normalize_answer({
        "answer": "Supported answer.",
        "claims": [{"claim": "Supported answer.", "status": "supported"}],
        "citations": [{"document": "FIR.pdf", "page": 2, "chunk_id": "FIR_p2_c4"}],
        "contradictions": ["None found."],
        "missing_information": ["Prior bail history not provided."],
        "confidence": "high",
    })
    assert result["answer"] == "Supported answer."
    assert result["claims"][0]["status"] == "supported"
    assert result["citations"][0]["chunk_id"] == "FIR_p2_c4"
    assert result["confidence"] == "high"
