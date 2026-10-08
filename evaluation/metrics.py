"""Evaluation metrics for grounded legal QA.

All metrics operate on supplied observations. They do not call an LLM and do
not infer success from missing data.
"""
from __future__ import annotations

import re
from typing import Any


def _norm(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def retrieval_recall_at_k(results: list[dict[str, Any]],
                          expected_sources: list[dict[str, Any]], k: int = 5) -> float:
    expected = {(x.get("document"), x.get("page"), x.get("chunk_id")) for x in expected_sources}
    if not expected:
        return 0.0
    found = {(x.get("document"), x.get("page"), x.get("chunk_id")) for x in results[:k]}
    return len(expected & found) / len(expected)


def citation_correctness(citations: list[dict[str, Any]],
                         expected_sources: list[dict[str, Any]]) -> float:
    expected = {(x.get("document"), x.get("page"), x.get("chunk_id")) for x in expected_sources}
    if not citations:
        return 0.0
    valid = 0
    for c in citations:
        key = (c.get("document"), c.get("page"), c.get("chunk_id"))
        valid += key in expected
    return valid / len(citations)


def claim_support(claims: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> float:
    if not claims:
        return 0.0
    evidence_tokens = _norm(" ".join(x.get("text", "") for x in evidence))
    supported = 0
    for claim in claims:
        text = claim if isinstance(claim, str) else claim.get("claim") or claim.get("text", "")
        tokens = _norm(text)
        supported += bool(tokens) and bool(tokens <= evidence_tokens)
    return supported / len(claims)


def groundedness(claims: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> float:
    """Percentage of supplied factual claims traceable to supplied evidence."""
    return claim_support(claims, evidence)


def fabricated_claim_rate(claims: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> float:
    return 1.0 - groundedness(claims, evidence) if claims else 0.0


def answer_usefulness(answer: str, expected_answer: str) -> float:
    expected = _norm(expected_answer)
    actual = _norm(answer)
    if not expected:
        return 0.0
    return len(expected & actual) / len(expected)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = ("groundedness", "retrieval_quality", "citation_correctness",
            "fabricated_claim_rate", "answer_usefulness")
    return {key: sum(float(r.get(key, 0.0)) for r in rows) / len(rows) for key in keys}
