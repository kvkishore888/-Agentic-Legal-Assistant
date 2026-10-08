"""Missing-information detection that never implies a fact does not exist."""
from __future__ import annotations
from typing import Any, Iterable, Mapping

DEFAULT_REQUIREMENTS = {
    "bail": ["accused name", "FIR number", "arrest date", "sections", "previous bail history", "medical report", "custody details"],
    "petition": ["court", "case number", "parties", "relief sought", "material facts", "supporting evidence"],
    "legal notice": ["sender", "recipient", "relevant agreement/event", "breach or grievance", "relief demanded", "date"],
    "affidavit": ["deponent name", "deponent address", "material facts", "verification details", "date", "place"],
}

def _corpus(hits: Iterable[Mapping[str, Any]]) -> str:
    return "\n".join(str(h.get("text", "")) for h in hits).lower()

def detect_missing_information(hits: Iterable[Mapping[str, Any]], required: Iterable[str]) -> list[dict[str, Any]]:
    corpus=_corpus(hits); missing=[]
    for item in required:
        label=str(item)
        if label.lower() not in corpus:
            missing.append({"item":label,"status":"NOT_FOUND_IN_PROVIDED_DOCUMENTS","message":f"{label}: not found in provided documents."})
    return missing

def detect_case_missing_information(hits: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return detect_missing_information(hits, ["accused name","FIR number","arrest date","sections","previous bail history","medical report","custody details"])

def requirements_for(document_type: str) -> list[str]:
    key=document_type.strip().lower()
    if key not in DEFAULT_REQUIREMENTS:
        raise ValueError(f"Unsupported document type: {document_type}")
    return list(DEFAULT_REQUIREMENTS[key])
