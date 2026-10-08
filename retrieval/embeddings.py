"""Lazy normalized embedding adapter."""
import os
class Embedder:
    def __init__(self,model_name=None): self.model_name=model_name or os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2"); self._model=None
    def _load(self):
        if self._model is None:
            try: from sentence_transformers import SentenceTransformer
            except ImportError as exc: raise RuntimeError("Install sentence-transformers for embeddings") from exc
            self._model=SentenceTransformer(self.model_name)
        return self._model
    def encode(self,texts):
        if not texts:return []
        return self._load().encode(texts,normalize_embeddings=True,show_progress_bar=False).tolist()