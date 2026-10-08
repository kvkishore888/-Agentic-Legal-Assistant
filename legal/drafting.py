"""Grounded drafting that uses placeholders rather than inventing facts."""
from __future__ import annotations
from .shared import retrieve,source_ref,get_grounding,IntegrationError
from .missing_info import detect_missing_information,requirements_for
from grounding.claim_extractor import extract_claims
def draft_document(document_type:str,query:str,*,top_k:int=5,required_information:list[str]|None=None)->dict:
    required=required_information or requirements_for(document_type); hits=retrieve(query,top_k)
    missing=detect_missing_information(hits,required)
    draft=[f"# {document_type.strip().upper()}","","## Verified source material"]
    draft += [f'- {h["document"]} — Page {h["page"]}: {h["text"]}' for h in hits] or ["- [NO VERIFIED SOURCE MATERIAL RETRIEVED]"]
    draft += ["","## Missing information"]+[f"- [MISSING — {m['item']}]" for m in missing]
    candidate=" ".join(h["text"] for h in hits)
    claims=extract_claims(candidate); verification=[]; citations=[]
    try:
        g=get_grounding(); verified=g.verify_claims(claims,hits)
        for c in verified:
            verification.append(c)
            if c.get("status") in {"SUPPORTED","PARTIALLY_SUPPORTED"} and c.get("chunk_id") is not None:
                citations.append({"claim_text":c.get("claim_text"),"document":c.get("document"),"page":c.get("page"),"chunk_id":c.get("chunk_id")})
        citations=g.validate_citations(citations,hits,verified)
    except IntegrationError as exc:
        verification=[{"status":"VERIFICATION_UNAVAILABLE","message":str(exc)}]
    safe=bool(verification) and all(c.get("status")=="SUPPORTED" for c in verification if "claim_text" in c) and not any(c.get("status")=="VERIFICATION_UNAVAILABLE" for c in verification)
    status="READY_FOR_REVIEW" if safe and not missing else "INCOMPLETE"
    return {"document_type":document_type,"status":status,"draft":"\n".join(draft),"missing_information":missing,"claim_verification":verification,"citations":citations,"sources":[source_ref(h) for h in hits]}
