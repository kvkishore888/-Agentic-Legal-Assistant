from grounding.citation_validator import validate_citations
from retrieval.keyword_search import KeywordIndex


def _chunk(chunk_id, text, document="case.pdf", page=1, content_type="LEGAL_EVIDENCE"):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "document": document,
        "page": page,
        "metadata": {"content_type": content_type},
    }


def test_keyword_search_respects_metadata_filter_and_excludes_eval_chunks():
    index = KeywordIndex()
    index.add([
        _chunk("legal-1", "The contract value is 240000."),
        _chunk("test-1", "Test questions: ask the chatbot questions about the contract.", content_type="EVALUATION_INSTRUCTIONS"),
        _chunk("other-1", "The contract value is 500000.", document="other.pdf"),
    ])
    result = index.search("contract value", top_k=10, where={"document": "case.pdf"})
    assert [x["chunk_id"] for x in result] == ["legal-1"]


def test_invalid_citation_cannot_become_valid_from_claim_text_substring():
    evidence = [_chunk("c1", "The contract value is 240000.")]
    claims = [{
        "claim_text": "The contract value is 999999.",
        "status": "UNSUPPORTED",
        "document": "case.pdf",
        "page": 1,
        "chunk_id": "c1",
    }]
    citations = [{
        "claim_text": "The contract value is 999999.",
        "document": "case.pdf",
        "page": 1,
        "chunk_id": "c1",
    }]
    result = validate_citations(citations, evidence, claims)
    assert result[0]["valid"] is False
    assert result[0]["checks"]["claim_verified"] is False


def test_verified_citation_requires_supported_claim():
    evidence = [_chunk("c1", "The contract value is 240000.")]
    claims = [{
        "claim_text": "The contract value is 240000.",
        "status": "SUPPORTED",
        "document": "case.pdf",
        "page": 1,
        "chunk_id": "c1",
    }]
    citations = [{
        "claim_text": "The contract value is 240000.",
        "document": "case.pdf",
        "page": 1,
        "chunk_id": "c1",
    }]
    result = validate_citations(citations, evidence, claims)
    assert result[0]["valid"] is True
