"""Embedding adapter. Uses sentence-transformers when installed."""
class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name=model_name
        self._model=None
    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers for embeddings") from exc
            self._model=SentenceTransformer(self.model_name)
        return self._model
    def encode(self, texts):
        return self._load().encode(texts, normalize_embeddings=True).tolist()
