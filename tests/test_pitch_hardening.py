from legal.missing_info import detect_missing_information
from legal.drafting import draft_document
from legal.research import research

H=[
 {"document":"FIR.pdf","page":2,"chunk_id":"a","text":"FIR No. 42/2026. The accused Ravi Kumar was arrested on 10 March 2026 under Section 420.","score":.9},
 {"document":"Bail_Order.pdf","page":4,"chunk_id":"b","text":"The accused Ravi Kumar was arrested on 12 March 2026 under Section 420.","score":.8},
]
def fake_retrieve(q,top_k=5): return H[:top_k]

def test_missing_info_uses_synonyms():
    missing=detect_missing_information(H,["medical report"])
    assert [x["item"] for x in missing]==["medical report"]

def test_drafting_is_explicitly_working_draft(monkeypatch):
    monkeypatch.setattr("legal.drafting.retrieve",fake_retrieve)
    r=draft_document("bail","arrest",required_information=["medical report"])
    assert r["status"]=="INCOMPLETE"
    assert "[MISSING — medical report]" in r["draft"]
    assert "not a filing-ready legal document" in r["draft"]

def test_research_does_not_claim_external_authority(monkeypatch):
    monkeypatch.setattr("legal.research.retrieve",fake_retrieve)
    r=research("arrest date")
    assert any("No independently verified external legal authority" in w for w in r["warnings"])
    assert all(x["source_type"]=="UPLOADED_CASE_EVIDENCE" for x in r["authorities"])
