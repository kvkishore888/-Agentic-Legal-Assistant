"""Optional cross-encoder reranker with safe fallback."""
import os
class Reranker:
    def __init__(self,model_name=None,enabled=True):
        self.model_name=model_name or os.getenv("RERANKER_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2"); self.enabled=enabled; self.model=None
    def rerank(self,query,results,top_k=5):
        if not results or top_k<=0:return []
        fallback=sorted(results,key=lambda r:r.get("fusion_score",r.get("score",0)),reverse=True)
        if not self.enabled:return fallback[:top_k]
        try:
            if self.model is None:
                from sentence_transformers import CrossEncoder
                self.model=CrossEncoder(self.model_name)
            scores=self.model.predict([(query,r["text"]) for r in fallback])
            ranked=[{**r,"rerank_score":float(s)} for r,s in zip(fallback,scores)]
            return sorted(ranked,key=lambda r:r["rerank_score"],reverse=True)[:top_k]
        except Exception:return fallback[:top_k]