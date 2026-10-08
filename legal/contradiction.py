"""Conservative contradiction detection with source evidence on both sides."""
from __future__ import annotations
import re
from typing import Any, Mapping, Sequence

CONTRADICTION = "CONTRADICTION"
POTENTIAL_CONTRADICTION = "POTENTIAL_CONTRADICTION"
NO_CONTRADICTION = "NO_CONTRADICTION"
_DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s+\d{4})\b")
_MONEY_RE = re.compile(r"(?:₹|Rs\.?|INR|\$|USD)\s?[\d,]+(?:\.\d+)?", re.I)
_SECTION_RE = re.compile(r"\b(?:Section|Sec\.?|S\.)\s*\d+[A-Za-z0-9()/-]*", re.I)

def _normalized(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _value_for(statement: str) -> tuple[str, str] | None:
    for label, pattern in (("date", _DATE_RE), ("amount", _MONEY_RE), ("section", _SECTION_RE)):
        match = pattern.search(statement)
        if match:
            return label, match.group(0).lower()
    return None

def compare_statements(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    a, b = str(left.get("text", "")), str(right.get("text", ""))
    va, vb = _value_for(a), _value_for(b)
    tokens_a = set(re.findall(r"[a-z]{4,}", _normalized(a)))
    tokens_b = set(re.findall(r"[a-z]{4,}", _normalized(b)))
    same_topic = len(tokens_a & tokens_b) >= 2
    status, reason = NO_CONTRADICTION, "No sufficiently supported conflict detected."
    if va and vb and va[0] == vb[0] and va[1] != vb[1] and same_topic:
        status, reason = CONTRADICTION, f"Both statements concern the same {va[0]} but state different values."
    elif same_topic and va and vb and va[1] != vb[1]:
        status, reason = POTENTIAL_CONTRADICTION, "The statements appear related but the available wording is insufficient to establish a direct contradiction."
    return {"status": status, "reason": reason, "statement_a": dict(left), "statement_b": dict(right)}

def find_contradictions(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings=[]
    for i, left in enumerate(evidence):
        for right in evidence[i+1:]:
            result=compare_statements(left, right)
            if result["status"] != NO_CONTRADICTION:
                findings.append(result)
    return findings
