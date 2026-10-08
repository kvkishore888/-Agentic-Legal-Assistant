"""Optional reranker with a safe lightweight default.

Cross-encoder reranking is opt-in because loading a transformer model can exceed
the memory limit of small cloud instances.
"""
from __future__ import annotations
import os

class Reranker:
    def __init__(self, model_name: str | None = None, enabled: bool | None = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        env_enabled = os.getenv("RERANKER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.enabled = env_enabled if enabled is None else enabled
        self.model = None

    def rerank(self, query, results, top_k=5):
        if not results or top_k <= 0:
            return []
        fallback = sorted(results, key=lambda r: r.get("fusion_score", r.get("score", 0)), reverse=True)
        if not self.enabled:
            return fallback[:top_k]
        try:
            if self.model is None:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
            scores = self.model.predict([(query, r["text"]) for r in fallback])
            ranked = [{**r, "rerank_score": float(s)} for r, s in zip(fallback, scores)]
            return sorted(ranked, key=lambda r: r["rerank_score"], reverse=True)[:top_k]
        except Exception:
            return fallback[:top_k]
