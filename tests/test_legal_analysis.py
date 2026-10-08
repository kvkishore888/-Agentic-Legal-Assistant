from legal.contradiction import CONTRADICTION,NO_CONTRADICTION,compare_statements
from legal.missing_info import detect_missing_information
from legal.case_review import review_case
from legal.drafting import draft_document
from legal.research import research
H=[{"document":"FIR.pdf","page":2,"chunk_id":"a","text":"Accused was arrested on 10 March 2026 under Section 420.","score":.9},{"document":"Bail_Order.pdf","page":4,"chunk_id":"b","text":"Accused was arrested on 12 March 2026 under Section 420.","score":.8}]
def fake_retrieve(q,top_k=5): return H[:top_k]
def test_contradiction(): assert compare_statements(H[0],H[1])["status"]==CONTRADICTION
def test_unrelated_statements(): assert compare_statements({"text":"Contract amount is Rs. 10,000."},{"text":"Meeting was on 12 March 2026."})["status"]==NO_CONTRADICTION
def test_missing_is_not_found():
    x=detect_missing_information(H,["medical report"]); assert x[0]["status"]=="NOT_FOUND_IN_PROVIDED_DOCUMENTS"
def test_review_preserves_sources(monkeypatch):
    monkeypatch.setattr("legal.case_review.retrieve",fake_retrieve); r=review_case("arrest"); assert r["key_facts"][0]["source"]["document"]=="FIR.pdf" and r["contradictions"]
def test_drafting_never_invents(monkeypatch):
    monkeypatch.setattr("legal.drafting.retrieve",fake_retrieve); r=draft_document("bail","arrest",required_information=["medical report"]); assert r["status"]=="INCOMPLETE" and "[MISSING — medical report]" in r["draft"]
def test_research_marks_unknown_authority_unverified(monkeypatch):
    monkeypatch.setattr("legal.research.retrieve",fake_retrieve); r=research("arrest date"); assert all(x["citation_status"]=="UNVERIFIED" for x in r["authorities"])
