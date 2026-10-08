"""Hybrid legal retrieval using weighted reciprocal-rank fusion and optional reranking."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from .embeddings import Embedder
from .vector_store import VectorStore
from .keyword_search import KeywordIndex
from .reranker import Reranker

class HybridRetriever:
    def __init__(self, store=None, embedder=None, reranker=None, vector_weight=1.0, keyword_weight=1.0, rrf_k=60, namespace=None):
        if vector_weight < 0 or keyword_weight < 0 or vector_weight + keyword_weight == 0:
            raise ValueError("Invalid retrieval weights")
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore(namespace=namespace)
        self.keyword = KeywordIndex()
        self.reranker = reranker or Reranker()
        self.vector_weight = float(vector_weight)
        self.keyword_weight = float(keyword_weight)
        self.rrf_k = int(rrf_k)

    def index(self, chunks):
        chunks = list(chunks)
        # Replace the in-memory lexical index for every indexing operation.
        # The vector collection is namespace-isolated by the active document set.
        self.keyword.add(chunks)
        if chunks:
            self.store.upsert(chunks, self.embedder.encode([c["text"] for c in chunks]))

    def retrieve(self, query, top_k=5, where=None):
        if top_k <= 0:
            return []
        candidate_k = max(top_k * 4, 20)
        vector = self.store.search(self.embedder.encode([query])[0], candidate_k, where)
        keyword = self.keyword.search(query, candidate_k, where)
        merged = {}

        def add(results, weight, source):
            for rank, item in enumerate(results, 1):
                cid = item["chunk_id"]
                entry = merged.setdefault(cid, dict(item))
                entry["fusion_score"] = entry.get("fusion_score", 0.0) + weight / (self.rrf_k + rank)
                entry.setdefault("retrieval_sources", [])
                if source not in entry["retrieval_sources"]:
                    entry["retrieval_sources"].append(source)

        add(vector, self.vector_weight, "vector")
        add(keyword, self.keyword_weight, "keyword")
        ordered = sorted(merged.values(), key=lambda r: r["fusion_score"], reverse=True)

        legal_only = [
            item for item in ordered
            if str((item.get("metadata") or {}).get("content_type", "LEGAL_EVIDENCE"))
            != "EVALUATION_INSTRUCTIONS"
        ]
        return self.reranker.rerank(query, legal_only, top_k)

_default = ContextVar("legal_retriever", default=None)

def configure(retriever):
    _default.set(retriever)

def retrieve(query, top_k=5):
    active = _default.get()
    if active is None:
        raise RuntimeError("Configure the shared retriever before calling retrieve()")
    return active.retrieve(query, top_k)

@contextmanager
def retrieval_scope(retriever):
    """Bind evidence to this request and restore the prior context on failure."""
    token = _default.set(retriever)
    try:
        yield
    finally:
        _default.reset(token)
