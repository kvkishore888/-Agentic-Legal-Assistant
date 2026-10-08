"""Case/contract review workflow."""
from __future__ import annotations
import re
from typing import Any
from .contradiction import find_contradictions
from .missing_info import detect_case_missing_information
from .shared import retrieve, source_ref

_DATE_RE=re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s+\d{4})\b")
_AMOUNT_RE=re.compile(r"(?:₹|Rs\.?|INR|\$|USD)\s?[\d,]+(?:\.\d+)?",re.I)
_SECTION_RE=re.compile(r"\b(?:Section|Sec\.?)\s*\d+[A-Za-z0-9()/-]*",re.I)

def _facts(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts=[]
    for hit in hits:
        text=hit["text"]; values=[]
        values += [("date",m.group(0)) for m in _DATE_RE.finditer(text)]
        values += [("amount",m.group(0)) for m in _AMOUNT_RE.finditer(text)]
        values += [("section",m.group(0)) for m in _SECTION_RE.finditer(text)]
        for kind,value in values:
            facts.append({"type":kind,"value":value,"source":source_ref(hit)})
        if not values:
            facts.append({"type":"statement","value":text.strip(),"source":source_ref(hit)})
    return facts

def review_case(query: str, top_k: int = 5) -> dict[str, Any]:
    hits=retrieve(query, top_k=top_k)
    facts=_facts(hits)
    evidence=[{"statement":h["text"],"source":source_ref(h)} for h in hits]
    contradictions=find_contradictions([{"text":h["text"],**source_ref(h)} for h in hits])
    missing=detect_case_missing_information(hits)
    documents=[]; seen=set()
    for hit in hits:
        doc=hit["document"]
        if doc not in seen:
            documents.append({"document":doc,"pages":[hit["page"]]}); seen.add(doc)
        else:
            for item in documents:
                if item["document"]==doc and hit["page"] not in item["pages"]:
                    item["pages"].append(hit["page"])
    return {"key_facts":facts,"evidence":evidence,"contradictions":contradictions,"missing_information":missing,"relevant_documents":documents}
