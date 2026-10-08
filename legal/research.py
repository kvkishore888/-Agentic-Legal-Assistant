"""Legal-authority research workflow connected to the current case."""
from __future__ import annotations
from typing import Any, Mapping
from .shared import get_grounding, retrieve, verification_ok, IntegrationError

def _authority(hit: Mapping[str,Any], relevance: float) -> dict[str,Any]:
    metadata=hit.get("metadata",{}) if isinstance(hit.get("metadata"),Mapping) else {}
    return {"case_name":metadata.get("case_name") or hit.get("case_name"),"court":metadata.get("court") or hit.get("court"),"date_year":metadata.get("date_year") or metadata.get("year") or hit.get("year"),"citation":metadata.get("citation") or hit.get("citation"),"relevant_passage":hit.get("text",""),"source_url_document":metadata.get("source_url") or metadata.get("source_document") or hit.get("document"),"relevance":relevance,"source_type":metadata.get("source_type") or hit.get("source_type") or "UNKNOWN","source":{"document":hit.get("document"),"page":hit.get("page"),"chunk_id":hit.get("chunk_id")}}

def research(query: str, case: Mapping[str,Any]|None=None, *, top_k:int=5) -> dict[str,Any]:
    context=[str(f["value"]) for f in (case or {}).get("key_facts",[]) if f.get("value")]
    hits=retrieve(query if not context else f"{query} {' '.join(context[:8])}", top_k=top_k)
    authorities=[]; warnings=[]
    try: grounding=get_grounding()
    except IntegrationError as exc: grounding=None; warnings.append(f"Citation verification unavailable: {exc}")
    for hit in hits:
        score=float(hit.get("score",0.0)) if isinstance(hit.get("score"),(int,float)) else 0.0
        item=_authority(hit,score); citation=item.get("citation")
        if grounding is None: item["citation_status"]="UNVERIFIED"
        elif not citation: item["citation_status"]="UNVERIFIED"; warnings.append(f"No citation metadata found for {item['source']['document']} page {item['source']['page']}.")
        else: item["citation_status"]="VERIFIED" if verification_ok(grounding.validate_citation(item)) else "UNVERIFIED"
        if item["source_type"]=="EXTERNAL_LEGAL_AUTHORITY" and item["citation_status"]=="VERIFIED": authorities.append(item)
        elif item["source_type"]!="EXTERNAL_LEGAL_AUTHORITY": item["source_type"]="UPLOADED_CASE_EVIDENCE"; authorities.append(item)
    return {"query":query,"authorities":authorities,"warnings":warnings,"case_context_used":context}
