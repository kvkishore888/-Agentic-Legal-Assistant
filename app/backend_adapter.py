"""Compatibility layer between the Member 4 UI and shared backend interfaces.

The adapter intentionally does not implement legal reasoning. It discovers the
team's shared retrieve/query/review interfaces when available and normalizes
their outputs for the UI.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

RETRIEVE_CANDIDATES = ("retrieval.hybrid_search", "retrieval", "backend.retrieval")
ANSWER_CANDIDATES = ("legal_analysis", "analysis", "backend.answer", "answer")
REVIEW_CANDIDATES = ("legal_analysis", "analysis", "backend.review")


def _find_function(candidates: tuple[str, ...], name: str) -> Callable[..., Any] | None:
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, name, None)
            if callable(fn):
                return fn
        except (ImportError, ModuleNotFoundError):
            continue
    return None


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    fn = _find_function(RETRIEVE_CANDIDATES, "retrieve")
    if fn is None:
        raise RuntimeError("Shared retrieve(query, top_k=5) interface is not available yet.")
    result = fn(query, top_k=top_k)
    return _normalize_evidence(result)


def answer_with_grounding(query: str, context: Any) -> Any:
    fn = _find_function(ANSWER_CANDIDATES, "answer_with_grounding")
    if fn is None:
        raise RuntimeError("Shared answer_with_grounding(query, context) interface is not available yet.")
    return fn(query, context)


def review_case(query: str) -> Any:
    fn = _find_function(REVIEW_CANDIDATES, "review_case")
    if fn is None:
        raise RuntimeError("Shared review_case(query) interface is not available yet.")
    return fn(query)


def _normalize_evidence(items: Any) -> list[dict[str, Any]]:
    if items is None:
        return []
    if isinstance(items, dict):
        items = items.get("evidence") or items.get("sources") or items.get("results") or [items]
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"text": item})
            continue
        if not isinstance(item, dict):
            continue
        normalized.append({
            "text": item.get("text") or item.get("content") or "",
            "document": item.get("document") or item.get("source") or item.get("file") or "Unknown",
            "page": item.get("page"),
            "chunk_id": item.get("chunk_id") or item.get("id"),
            "score": item.get("score"),
            **item,
        })
    return normalized


def normalize_answer(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"answer": raw, "claims": [], "citations": [], "contradictions": [],
                "missing_information": [], "confidence": None, "evidence": []}
    if not isinstance(raw, dict):
        return {"answer": str(raw), "claims": [], "citations": [], "contradictions": [],
                "missing_information": [], "confidence": None, "evidence": []}
    return {
        "answer": raw.get("answer") or raw.get("response") or raw.get("text") or "",
        "claims": raw.get("claims") or [],
        "citations": raw.get("citations") or raw.get("sources") or [],
        "contradictions": raw.get("contradictions") or [],
        "missing_information": raw.get("missing_information") or raw.get("missing") or [],
        "confidence": raw.get("confidence"),
        "evidence": _normalize_evidence(raw.get("evidence") or raw.get("sources") or []),
        **raw,
    }
