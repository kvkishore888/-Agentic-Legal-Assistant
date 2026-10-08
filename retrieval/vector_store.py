"""Persistent Chroma adapter preserving source metadata and supporting metadata filters."""
from __future__ import annotations
import json
import os
from pathlib import Path

class VectorStore:
    def __init__(self, path=None, collection=None):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb for vector storage") from exc
        db_path = path or os.getenv("VECTOR_DB_PATH", ".legal_chroma")
        collection_name = collection or os.getenv("VECTOR_COLLECTION", "legal_chunks")
        self.client = chromadb.PersistentClient(path=str(Path(db_path)))
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _clean(meta):
        out = {}
        for key, value in (meta or {}).items():
            if value is None:
                continue
            out[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value, ensure_ascii=False)
        return out

    def upsert(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self.collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[self._clean({**(c.get("metadata") or {}), "document": c["document"], "page": c["page"], "section": c.get("section", ""), "chunk_id": c["chunk_id"]}) for c in chunks],
        )

    def search(self, query_embedding, top_k=5, where=None):
        if top_k <= 0:
            return []
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where, include=["documents", "metadatas", "distances"])
        rows = []
        for text, meta, chunk_id, distance in zip((result.get("documents") or [[]])[0], (result.get("metadatas") or [[]])[0], (result.get("ids") or [[]])[0], (result.get("distances") or [[]])[0]):
            metadata = dict(meta or {})
            rows.append({"text": text, "document": metadata.get("document", ""), "page": metadata.get("page"), "section": metadata.get("section", ""), "chunk_id": chunk_id, "score": 1 - float(distance), "metadata": metadata})
        return rows
