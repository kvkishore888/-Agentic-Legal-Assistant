"""Optional cross-encoder reranker."""
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name=model_name; self.model=None
    def rerank(self,query,results,top_k=5):
        if not results:return []
        try:
            if self.model is None:
                from sentence_transformers import CrossEncoder
                self.model=CrossEncoder(self.model_name)
            scores=self.model.predict([(query,r["text"]) for r in results])
            ranked=[{**r,"score":float(s)} for r,s in zip(results,scores)]
            return sorted(ranked,key=lambda x:x["score"],reverse=True)[:top_k]
        except ImportError:
            return sorted(results,key=lambda x:x.get("score",0),reverse=True)[:top_k]
