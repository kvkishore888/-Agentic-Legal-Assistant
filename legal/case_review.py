"""Source-preserving case/contract review workflow."""
from __future__ import annotations
import re
from .shared import retrieve,source_ref
from .contradiction import find_contradictions
from .missing_info import detect_case_missing_information
_PATTERNS=(("date",r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s+\d{4})\b"),("amount",r"(?:₹|Rs\.?|INR|\$|USD)\s?[\d,]+(?:\.\d+)?"),("section",r"\b(?:Section|Sec\.?)\s*\d+[A-Za-z0-9()/-]*"))
def _facts(hits):
    facts=[]
    for h in hits:
        found=False
        for kind,pat in _PATTERNS:
            for m in re.finditer(pat,h["text"],re.I): facts.append({"type":kind,"value":m.group(0),"source":source_ref(h)}); found=True
        if not found:facts.append({"type":"statement","value":h["text"].strip(),"source":source_ref(h)})
    return facts
def review_case(query:str,top_k:int=5)->dict:
    hits=retrieve(query,top_k); facts=_facts(hits); evidence=[{"statement":h["text"],"source":source_ref(h)} for h in hits]
    contradictions=find_contradictions([{**source_ref(h),"text":h["text"]} for h in hits])
    docs=[]
    for h in hits:
        d=next((x for x in docs if x["document"]==h["document"]),None)
        if not d:d={"document":h["document"],"pages":[]}; docs.append(d)
        if h["page"] not in d["pages"]:d["pages"].append(h["page"])
    return {"key_facts":facts,"evidence":evidence,"contradictions":contradictions,"missing_information":detect_case_missing_information(hits),"relevant_documents":docs}
