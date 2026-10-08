"""Deterministic atomic claim extraction."""
from __future__ import annotations
import re
from typing import Any
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
def _clean(text: str) -> str: return re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n-•")
def extract_claims(answer: str) -> list[dict[str, Any]]:
    if not isinstance(answer, str) or not answer.strip(): return []
    claims=[]
    for raw in _SENTENCE_RE.split(answer):
        text=_clean(raw)
        if not text or text.rstrip(".")=="Not verified from the provided sources": continue
        parts=re.split(r"(?i)\s+(?:and|but)\s+(?=(?:the|a|an|he|she|they|it|accused|defendant|contract|court|released|arrested|dismissed|granted|paid)\b)",text)
        for part in parts:
            claim=_clean(part)
            if len(claim.split())>=2:
                claims.append({"claim_text":claim,"status":"UNCERTAIN","supporting_evidence":[],"document":None,"page":None,"chunk_id":None,"confidence":0.0})
    return claims
