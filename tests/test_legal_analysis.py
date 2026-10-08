from legal.case_review import review_case
from legal.contradiction import CONTRADICTION, NO_CONTRADICTION, compare_statements
from legal.drafting import draft_document
from legal.missing_info import detect_missing_information
from legal.research import research

HITS=[{"document":"FIR.pdf","page":2,"chunk_id":"a","text":"Accused was arrested on 10 March 2026 under Section 420.","score":0.9},{"document":"Bail_Order.pdf","page":4,"chunk_id":"b","text":"Accused was arrested on 12 March 2026 under Section 420.","score":0.8}]

def fake_retrieve(query, top_k=5): return HITS[:top_k]

def test_contradiction_requires_same_topic():
    assert compare_statements(HITS[0],HITS[1])["status"]==CONTRADICTION
    assert compare_statements({"text":"The contract amount is Rs. 10,000."},{"text":"The meeting was on 12 March 2026."})["status"]==NO_CONTRADICTION

def test_missing_means_not_found_not_nonexistent():
    missing=detect_missing_information(HITS,["accused name","medical report"])
    assert all(x["status"]=="NOT_FOUND_IN_PROVIDED_DOCUMENTS" for x in missing)

def test_review_preserves_source(monkeypatch):
    monkeypatch.setattr("legal.case_review.retrieve",fake_retrieve)
    result=review_case("arrest")
    assert result["key_facts"][0]["source"]["document"]=="FIR.pdf"
    assert result["key_facts"][0]["source"]["page"]==2
    assert result["contradictions"][0]["status"]==CONTRADICTION

def test_drafting_never_claims_ready_without_grounding(monkeypatch):
    monkeypatch.setattr("legal.drafting.retrieve",fake_retrieve)
    monkeypatch.setattr("legal.drafting.review_case",lambda q,top_k=5:{"key_facts":[],"evidence":[],"contradictions":[],"missing_information":[],"relevant_documents":[]})
    result=draft_document("bail","arrest",required_information=[])
    assert result["status"]=="INCOMPLETE"
    assert "VERIFICATION_UNAVAILABLE" in str(result["claim_verification"])

def test_research_does_not_call_unverified_authority_authoritative(monkeypatch):
    monkeypatch.setattr("legal.research.retrieve",fake_retrieve)
    result=research("arrest date")
    assert all(x["citation_status"]=="UNVERIFIED" for x in result["authorities"])
