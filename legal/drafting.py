"""Grounded legal drafting templates; no invented case facts or authorities."""
from __future__ import annotations
from typing import Any, Mapping
from .case_review import review_case
from .missing_info import detect_missing_information, requirements_for
from .shared import get_grounding, retrieve, source_ref, verification_ok, IntegrationError

def _value_map(review: Mapping[str, Any]) -> dict[str,str]:
    values={}
    for fact in review.get("key_facts",[]):
        if fact.get("type") in {"date","amount","section"}:
            values.setdefault(fact["type"],str(fact["value"]))
    return values

def draft_document(document_type: str, query: str, *, top_k: int=5, required_information: list[str]|None=None) -> dict[str,Any]:
    required=required_information or requirements_for(document_type)
    hits=retrieve(query, top_k=top_k)
    missing=detect_missing_information(hits, required)
    review=review_case(query, top_k=top_k)
    values=_value_map(review)
    draft=[f"# {document_type.strip().upper()}","","## Verified case material",""]
    draft.extend([f'- {h["document"]} — Page {h["page"]}: {h["text"]}' for h in hits] or ["[No source-backed material was retrieved.]"])
    draft += ["","## Drafting placeholders",""]
    draft += [f"- [MISSING — {item['item']}]" for item in missing] or ["- Required information was found in the retrieved corpus; claim verification is still required before filing."]
    draft += ["","## Structured details",""] + [f"- {k}: {v}" for k,v in values.items()]
    verification=[]
    try:
        grounding=get_grounding()
        for fact in review["key_facts"]:
            result=grounding.verify_claim(fact["value"],fact["source"])
            verification.append({"claim":fact["value"],"verified":verification_ok(result),"source":fact["source"]})
    except IntegrationError as exc:
        verification.append({"verified":False,"status":"VERIFICATION_UNAVAILABLE","message":str(exc)})
    claim_items=[v for v in verification if "claim" in v]
    all_verified=bool(verification) and bool(claim_items) and all(v["verified"] for v in claim_items) and not any(v.get("status")=="VERIFICATION_UNAVAILABLE" for v in verification)
    status="READY_FOR_REVIEW" if not missing and all_verified else "INCOMPLETE"
    return {"document_type":document_type,"status":status,"draft":"\n".join(draft),"missing_information":missing,"claim_verification":verification,"sources":[source_ref(h) for h in hits]}
