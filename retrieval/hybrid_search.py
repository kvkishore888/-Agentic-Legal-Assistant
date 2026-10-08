"""Hybrid retrieval combining vector and keyword evidence."""
from .embeddings import Embedder
from .vector_store import VectorStore
from .keyword_search import KeywordIndex
from .reranker import Reranker

class HybridRetriever:
    def __init__(self,store=None,embedder=None,reranker=None):
        self.embedder=embedder or Embedder(); self.store=store or VectorStore()
        self.keyword=KeywordIndex(); self.reranker=reranker or Reranker()
    def index(self,chunks):
        self.keyword.add(chunks); self.store.upsert(chunks,self.embedder.encode([c["text"] for c in chunks]))
    def retrieve(self,query,top_k=5,where=None):
        vector=self.store.search(self.embedder.encode([query])[0],max(top_k*2,10),where)
        keyword=self.keyword.search(query,max(top_k*2,10))
        merged={}
        for r in vector+keyword:
            merged.setdefault(r["chunk_id"],r)
            if r["chunk_id"] in merged: merged[r["chunk_id"]]["score"]=max(merged[r["chunk_id"]]["score"],r["score"])
        return self.reranker.rerank(query,list(merged.values()),top_k)

_default=None
def configure(retriever): 
    global _default; _default=retriever
def retrieve(query,top_k=5):
    if _default is None: raise RuntimeError("Configure the shared retriever before calling retrieve()")
    return _default.retrieve(query,top_k)
