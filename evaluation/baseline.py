"""Basic RAG baseline: document -> chunks -> vector retrieval -> answer.

The answer generator is injected so evaluation never silently invents an LLM
or results. A production caller can provide the team's existing grounded
answer function.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_documents
from retrieval.embeddings import Embedder
from retrieval.vector_store import VectorStore


class BasicRAG:
    def __init__(self, answer_fn: Callable[[str, list[dict[str, Any]]], Any],
                 embedder: Embedder | None = None, store: VectorStore | None = None):
        self.answer_fn = answer_fn
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

    def index(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.store.upsert(chunks, self.embedder.encode([c["text"] for c in chunks]))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = self.store.search(self.embedder.encode([query])[0], top_k)
        by_id = {c["chunk_id"]: c for c in self.chunks}
        return [{**by_id.get(r["chunk_id"], {}), **r} for r in results]

    def answer(self, query: str, top_k: int = 5) -> dict[str, Any]:
        context = self.retrieve(query, top_k)
        raw = self.answer_fn(query, context)
        return {"answer": raw.get("answer") if isinstance(raw, dict) else str(raw),
                "evidence": context, "raw": raw}


def load_text_fixture(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return chunk_pages([{"text": text, "document": p.name, "page": 1}])
