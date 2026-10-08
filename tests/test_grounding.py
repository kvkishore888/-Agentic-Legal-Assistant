from grounding.claim_extractor import extract_claims
from grounding.claim_verifier import verify_claims
from grounding.citation_validator import validate_citations
from agent.agent import answer_with_grounding
from agent.router import route
E=[{"text":"Accused was arrested on 10 March.","document":"FIR.pdf","page":2,"chunk_id":"FIR_p2_c0","score":.91}]
def test_extracts_atomic_claims(): assert len(extract_claims("The accused was arrested on 10 March and released on 15 March."))==2
def test_supported_exact_evidence(): assert verify_claims(extract_claims("Accused was arrested on 10 March."),E)[0]["status"]=="SUPPORTED"
def test_wrong_date_is_unsupported(): assert verify_claims(extract_claims("The accused was arrested on 15 March."),E)[0]["status"]=="UNSUPPORTED"
def test_negated_claim_is_contradicted(): assert verify_claims(extract_claims("The accused was not arrested on 10 March."),E)[0]["status"]=="CONTRADICTED"
def test_no_evidence_uncertain(): assert verify_claims(extract_claims("The court dismissed the petition."),[])[0]["status"]=="UNCERTAIN"
def test_numeric_conflict_is_unsupported():
    e=[{"text":"The amount involved was 50000 rupees.","document":"FIR.pdf","page":2,"chunk_id":"x"}]
    assert verify_claims(extract_claims("The amount involved was 500000 rupees."),e)[0]["status"]=="UNSUPPORTED"
def test_citation_requires_real_chunk(): assert validate_citations([{"claim_text":"x","document":"FIR.pdf","page":99,"chunk_id":"missing"}],E)[0]["valid"] is False
def test_citation_accepts_verified_paraphrase():
    c={"claim_text":"The accused was arrested on 10 March.","document":"FIR.pdf","page":2,"chunk_id":"FIR_p2_c0"}
    assert validate_citations([c],E,[{**c,"status":"SUPPORTED"}])[0]["valid"] is True
def test_grounding_rejects_unsupported_claim():
    out=answer_with_grounding("When?",{"answer":"The accused was arrested on 15 March.","evidence":E})
    assert out["answer"]=="Not verified from the provided sources." and out["confidence"]=="LOW"
def test_grounding_accepts_supported_claim():
    out=answer_with_grounding("When?",{"answer":"The accused was arrested on 10 March.","evidence":E})
    assert out["citations"][0]["valid"] is True and out["confidence"]=="HIGH"
def test_router():
    assert route("Review this contract clause")=="CASE_CONTRACT_REVIEW"
    assert route("Draft a legal notice")=="LEGAL_DRAFTING"
    assert route("Research precedent on limitation")=="LEGAL_RESEARCH"
    assert route("What does this FIR say?")=="GROUNDED_RAG_CHAT"
