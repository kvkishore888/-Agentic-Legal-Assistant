from grounding.claim_extractor import extract_claims
from grounding.claim_verifier import verify_claims
from grounding.citation_validator import validate_citations
from agent.agent import answer_with_grounding
from agent.router import route
E=[{"text":"Accused was arrested on 10 March.","document":"FIR.pdf","page":2,"chunk_id":"FIR_p2_c0","score":0.91}]
def test_extracts_atomic_claims(): assert len(extract_claims("The accused was arrested on 10 March and released on 15 March."))==2
def test_supported_exact_evidence():
    v=verify_claims(extract_claims("Accused was arrested on 10 March."),E); assert v[0]["status"]=="SUPPORTED"; assert v[0]["document"]=="FIR.pdf" and v[0]["page"]==2
def test_wrong_date_not_supported():
    v=verify_claims(extract_claims("The accused was arrested on 15 March."),E); assert v[0]["status"] in {"UNSUPPORTED","CONTRADICTED"}
def test_no_evidence_uncertain(): assert verify_claims(extract_claims("The court dismissed the petition."),[])[0]["status"]=="UNCERTAIN"
def test_citation_requires_real_chunk(): assert validate_citations([{"claim_text":"x","document":"FIR.pdf","page":99,"chunk_id":"missing"}],E)[0]["valid"] is False
def test_grounding_rejects_unsupported_claim():
    out=answer_with_grounding("When?",{"answer":"The accused was arrested on 15 March.","evidence":E}); assert out["answer"]=="Not verified from the provided sources." and out["confidence"]=="LOW"
def test_grounding_accepts_supported_claim():
    out=answer_with_grounding("When?",{"answer":"The accused was arrested on 10 March.","evidence":E}); assert out["answer"]=="The accused was arrested on 10 March." and out["citations"][0]["valid"] is True and out["confidence"]=="HIGH"
def test_router():
    assert route("Review this contract clause")=="CASE_CONTRACT_REVIEW"; assert route("Draft a legal notice")=="LEGAL_DRAFTING"; assert route("Research precedent on limitation")=="LEGAL_RESEARCH"; assert route("What does this FIR say?")=="GROUNDED_RAG_CHAT"
