"""Shared adapters for retrieval, grounding, and source-preserving findings."""
from __future__ import annotations
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

RetrieveFn = Callable[..., Iterable[Mapping[str, Any]]]
VerifyFn = Callable[..., Any]

class IntegrationError(RuntimeError):
    """Raised when a required shared service is unavailable or malformed."""

def get_retriever() -> RetrieveFn:
    try:
        module = importlib.import_module("retrieval.hybrid_search")
        fn = getattr(module, "retrieve", None)
    except ImportError as exc:
        raise IntegrationError("Member 1 retrieval is not available") from exc
    if not callable(fn):
        raise IntegrationError("retrieval.hybrid_search.retrieve is not configured")
    return fn

@dataclass(frozen=True)
class GroundingService:
    verify_claim: VerifyFn
    validate_citation: VerifyFn

_grounding: GroundingService | None = None

def configure_grounding(service: GroundingService) -> None:
    global _grounding
    _grounding = service

def get_grounding() -> GroundingService:
    if _grounding is not None:
        return _grounding
    candidates = (
        ("grounding.claim_verification", "verify_claim", "validate_citation"),
        ("grounding.verification", "verify_claim", "validate_citation"),
        ("grounding.claim_verifier", "verify_claim", "validate_citation"),
    )
    for module_name, claim_name, citation_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        claim = getattr(module, claim_name, None)
        citation = getattr(module, citation_name, None)
        if callable(claim) and callable(citation):
            return GroundingService(claim, citation)
    raise IntegrationError("Member 2 claim/citation verification is not configured")

def normalize_result(item: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError("retrieval result must be a mapping")
    text = str(item.get("text", ""))
    if not text.strip():
        raise ValueError("retrieval result has no text")
    result = dict(item)
    result["document"] = str(item.get("document", "UNKNOWN"))
    result["page"] = item.get("page", "UNKNOWN")
    result["chunk_id"] = item.get("chunk_id", "UNKNOWN")
    result["text"] = text
    if "score" in item:
        try:
            result["score"] = float(item["score"])
        except (TypeError, ValueError):
            pass
    result["source"] = f'{result["document"]} — Page {result["page"]}'
    return result

def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    raw = get_retriever()(query, top_k=top_k)
    return [normalize_result(x) for x in raw]

def verification_ok(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, Mapping):
        return bool(result.get("verified", False))
    return False

def source_ref(hit: Mapping[str, Any]) -> dict[str, Any]:
    return {"document": hit.get("document", "UNKNOWN"), "page": hit.get("page", "UNKNOWN"), "chunk_id": hit.get("chunk_id", "UNKNOWN"), "source": hit.get("source", "UNKNOWN"), "text": hit.get("text", "")}
