"""Deterministic guard that keeps the assistant focused on the uploaded legal case."""
from __future__ import annotations
import re
from typing import Any

# These are intentionally high-signal, non-legal patterns. Legal facts such as
# "murder", "assault", "crime", etc. are NOT blocked because they may be central
# to a case.
_OFF_TOPIC_PATTERNS = {
    "general_chat": r"\b(hello|hi|hey|how are you|good morning|good night|tell me a joke|make me laugh)\b",
    "food": r"\b(recipe|pizza|burger|biryani|cooking|food|restaurant|diet|calories)\b",
    "sports": r"\b(cricket|football|soccer|basketball|tennis|ipl|match score|sports news)\b",
    "entertainment": r"\b(movie|movies|song|songs|music|anime|netflix|actor|actress|celebrity)\b",
    "technology": r"\b(programming|python|javascript|java|c\+\+|html|css|coding|debug my code|computer science)\b",
    "personal": r"\b(girlfriend|boyfriend|relationship advice|love advice|dating|my horoscope|astrology)\b",
    "travel": r"\b(weather|flight|hotel|tourist|travel plan|trip plan|places to visit)\b",
}

def _case_signals(query: str) -> int:
    q = query.lower()
    signals = (
        r"\b(case|contract|agreement|clause|claim|complaint|petition|notice|affidavit|bail|court|judge|plaintiff|defendant|appellant|respondent)\b",
        r"\b(evidence|document|witness|testimony|hearing|filing|deadline|payment|delivery|breach|damages|liability|dispute)\b",
        r"\b(section|act|statute|judgment|precedent|citation|legal|law|rights|obligation)\b",
        r"\b(this document|this case|these documents|uploaded file|provided source|provided document)\b",
    )
    return sum(bool(re.search(pattern, q)) for pattern in signals)

def guard_query(query: str) -> dict[str, Any]:
    q = str(query or "").strip()
    if not q:
        return {
            "allowed": False,
            "reason": "EMPTY_QUERY",
            "message": "Please ask a question about the uploaded case or legal documents.",
            "suggestions": _suggestions(),
        }

    # Legal/case signals take precedence over incidental words such as "food"
    # appearing inside a quoted document or factual description.
    if _case_signals(q) > 0:
        return {"allowed": True, "reason": "CASE_RELEVANT", "message": "", "suggestions": []}

    for category, pattern in _OFF_TOPIC_PATTERNS.items():
        if re.search(pattern, q):
            return {
                "allowed": False,
                "reason": "OFF_TOPIC",
                "category": category,
                "message": (
                    "This question is outside the scope of the uploaded legal case. "
                    "Please ask something related to the case, contract, evidence, "
                    "documents, legal issues, or drafting needs."
                ),
                "suggestions": _suggestions(),
            }

    # For ambiguous questions, allow them through. Retrieval/grounding remains
    # the final evidence gate, so an unfamiliar legal query is not rejected just
    # because it lacks one of the simple keywords above.
    return {"allowed": True, "reason": "UNCLASSIFIED", "message": "", "suggestions": []}

def _suggestions() -> list[str]:
    return [
        "What are the key facts of this case?",
        "What evidence supports the main claim?",
        "Are there any contradictions in the documents?",
        "What important information is missing from the case?",
        "What legal issues or clauses should I review?",
    ]
