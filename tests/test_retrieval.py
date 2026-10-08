from retrieval.keyword_search import KeywordIndex
from retrieval.reranker import Reranker
from retrieval.hybrid_search import HybridRetriever

def test_keyword_retrieval():
    index=KeywordIndex()
    index.add([{"text":"court granted bail to accused","chunk_id":"a","document":"case.pdf","page":1},
               {"text":"payment terms","chunk_id":"b","document":"contract.pdf","page":2}])
    assert index.search("bail accused",1)[0]["chunk_id"]=="a"

class FakeEmbedder:
    def encode(self,texts): return [[1.0,0.0] for _ in texts]

class FakeStore:
    def __init__(self,rows): self.rows=rows
    def upsert(self,*args): pass
    def search(self,*args,**kwargs): return self.rows

class NoopReranker:
    def rerank(self,q,rows,top_k=5): return sorted(rows,key=lambda x:x["fusion_score"],reverse=True)[:top_k]

def test_rrf_fusion_and_metadata():
    chunks=[{"text":"Bail was granted","chunk_id":"a","document":"case.pdf","page":3,"section":"Order","metadata":{"source_type":"case"}},
            {"text":"Payment is due","chunk_id":"b","document":"contract.pdf","page":2,"section":"Payment","metadata":{"source_type":"contract"}}]
    h=HybridRetriever(store=FakeStore([dict(chunks[0],score=.9)]),embedder=FakeEmbedder(),reranker=NoopReranker())
    h.keyword.add(chunks)
    rows=h.retrieve("bail",2)
    assert rows[0]["chunk_id"]=="a"
    assert rows[0]["metadata"]["source_type"]=="case"
    assert rows[0]["fusion_score"]>0

def test_reranker_runtime_failure_falls_back():
    class Broken:
        def predict(self,pairs): raise RuntimeError("model unavailable")
    r=Reranker(); r.model=Broken()
    assert r.rerank("q",[{"text":"evidence","chunk_id":"x","fusion_score":1.0}],1)[0]["chunk_id"]=="x"
