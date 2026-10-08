from retrieval.embeddings import Embedder
from retrieval.reranker import Reranker

def test_default_embedder_is_lightweight(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    embedder = Embedder()
    vectors = embedder.encode(["court granted bail", "payment is due"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 256
    assert abs(sum(x * x for x in vectors[0]) - 1.0) < 1e-6

def test_reranker_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)
    reranker = Reranker()
    assert reranker.enabled is False
