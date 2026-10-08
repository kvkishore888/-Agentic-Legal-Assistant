from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
from retrieval.hybrid_search import retrieval_scope, retrieve
from app.backend_adapter import run_workflow

class Fake:
    def __init__(self, name): self.name = name
    def retrieve(self, query, top_k=5):
        return [{"document":self.name,"page":1,"chunk_id":self.name,"text":"The contract value is 240000 rupees."}]

def test_concurrent_workspaces_do_not_share_evidence():
    barrier = Barrier(2)
    def request(name):
        with retrieval_scope(Fake(name)):
            barrier.wait()
            return run_workflow("Grounded RAG Chat", "contract value")["evidence"][0]["document"]
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(request, ["alice.pdf", "bob.pdf"])) == ["alice.pdf", "bob.pdf"]

def test_scope_restored_after_failure():
    with retrieval_scope(Fake("alice.pdf")):
        with pytest.raises(RuntimeError):
            with retrieval_scope(Fake("bob.pdf")):
                raise RuntimeError("failed request")
        assert retrieve("value")[0]["document"] == "alice.pdf"

def test_unknown_workflow_rejected():
    with pytest.raises(ValueError, match="Unsupported workflow"):
        run_workflow("typo", "value")


def test_supported_claim_cannot_validate_another_claim_in_same_chunk():
    from grounding.citation_validator import validate_citations
    source = {"document":"a.pdf","page":1,"chunk_id":"a","text":"The amount is 100 rupees."}
    supported = {**source,"claim_text":"The amount is 100 rupees.","status":"SUPPORTED"}
    forged = {**source,"claim_text":"The amount is 999 rupees."}
    assert not validate_citations([forged], [source], [supported])[0]["valid"]
