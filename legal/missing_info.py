"""Detect missing material without claiming the material does not exist."""
from __future__ import annotations
import re
from typing import Iterable, Mapping

DEFAULT_REQUIREMENTS = {
    "bail": ["accused name", "FIR number", "arrest date", "sections", "previous bail history", "medical report", "custody details"],
    "petition": ["court", "case number", "parties", "relief sought", "material facts", "supporting evidence"],
    "legal notice": ["sender", "recipient", "relevant agreement/event", "breach or grievance", "relief demanded", "date"],
    "affidavit": ["deponent name", "deponent address", "material facts", "verification details", "date", "place"],
    "case review": ["parties", "case number", "material facts", "dates/events", "supporting evidence"],
}

_SYNONYMS = {
    "parties": (r"\b(?:plaintiff|petitioner|appellant|complainant|defendant|respondent|accused|party|parties)\b",),
    "case number": (r"\b(?:case|fir|crime|petition|appeal|writ)\s*(?:no\.?|number)\s*[:\-]?\s*[A-Za-z0-9/.-]+",),
    "material facts": (r"\b(?:facts|background|brief facts|material facts|facts of the case)\b",),
    "dates/events": (r"\b(?:date|dated|on|event|incident|arrested|executed|signed)\b",),
    "supporting evidence": (r"\b(?:evidence|exhibit|annexure|document|FIR|order|report|record)\b",),
    "sections": (r"\b(?:section|sec\.?|sections)\s*\d",),
    "court": (r"\b(?:court|tribunal|high court|supreme court|district court)\b",),
    "relief sought": (r"\b(?:relief|prayer|prayers|seek|sought|requested)\b",),
    "previous bail history": (r"\b(?:previous|earlier|prior)\s+bail\b|\bbail\s+history\b",),
    "medical report": (r"\bmedical\b.{0,40}\b(?:report|certificate|record)\b",),
    "custody details": (r"\b(?:custody|custodial|remand)\b",),
    "accused name": (r"\baccused\b.{0,50}\b(?:name|named|called)\b|\bname\s+of\s+(?:the\s+)?accused\b",),
    "FIR number": (r"\bFIR\s*(?:no\.?|number)\s*[:\-]?\s*[A-Za-z0-9/.-]+",),
    "arrest date": (r"\barrest(?:ed)?\b.{0,35}\b(?:on|dated)\b|\barrest\s+date\b"),
    "sender": (r"\bsender\b|\bfrom\b"),
    "recipient": (r"\brecipient\b|\bto\b"),
    "relevant agreement/event": (r"\b(?:agreement|contract|event|transaction|incident)\b"),
    "breach or grievance": (r"\b(?:breach|default|grievance|complaint|violation|non[- ]payment)\b"),
    "relief demanded": (r"\b(?:relief|demand|compensation|payment|remedy|cease)\b"),
    "deponent name": (r"\bdeponent\b.{0,50}\b(?:name|named)\b|\bname\s+of\s+(?:the\s+)?deponent\b"),
    "deponent address": (r"\bdeponent\b.{0,80}\baddress\b|\baddress\b.{0,80}\bdeponent\b"),
    "verification details": (r"\bverification\b|\bverified\b"),
    "date": (r"\b(?:date|dated|on)\b"),
    "place": (r"\bplace\b|\bat\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b"),
}

def _present(corpus: str, item: str) -> bool:
    label = item.lower()
    patterns = _SYNONYMS.get(label, (re.escape(label),))
    if isinstance(patterns, str):
        patterns = (patterns,)
    return any(re.search(p, corpus, re.I | re.S) for p in patterns)

def detect_missing_information(hits: Iterable[Mapping], required: Iterable[str]) -> list[dict]:
    corpus = "\n".join(str(h.get("text", "")) for h in hits).strip()
    out = []
    for item in required:
        label = str(item)
        if not _present(corpus, label):
            out.append({"item": label, "status": "NOT_FOUND_IN_PROVIDED_DOCUMENTS",
                        "message": f"{label}: not found in provided documents."})
    return out

def requirements_for(document_type: str) -> list[str]:
    key = document_type.strip().lower()
    if key not in DEFAULT_REQUIREMENTS:
        raise ValueError(f"Unsupported document type: {document_type}")
    return list(DEFAULT_REQUIREMENTS[key])

def detect_case_missing_information(hits):
    return detect_missing_information(hits, DEFAULT_REQUIREMENTS["case review"])
