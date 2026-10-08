"""Lightweight deterministic embeddings with optional external model support.

The default is hash-based embeddings so the Streamlit/Render deployment does not
need PyTorch or a multi-hundred-MB model just to index a document.
"""
from __future__ import annotations
import hashlib
import math
import os
import re

_TOKEN_RE = re.compile(r"(?u)\b\w+\b")

class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "hash")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "256"))
        self._model = None
        if self.dimension < 32:
            raise ValueError("EMBEDDING_DIMENSION must be >= 32")

    def _load(self):
        if self.model_name.lower() in {"hash", "local", "lightweight"}:
            return None
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Optional model embeddings require sentence-transformers; use EMBEDDING_MODEL=hash on small deployments."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _hash_encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(str(text).lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def encode(self, texts):
        if not texts:
            return []
        model = self._load()
        if model is None:
            return [self._hash_encode(text) for text in texts]
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
