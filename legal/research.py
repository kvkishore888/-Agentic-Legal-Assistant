"""Retrieval-connected legal research; never upgrades unknown metadata to authoritative."""
from __future__ import annotations
from typing import Mapping,Any
from .shared import retrieve,get_grounding,IntegrationError
def research(query:str,case:Mapping[str,Any]|None=None,*,top_k:int=5)->dict[str,Any]:
    context=[str(x.get("value")) for x in (case or {}).get("key_facts",[]) if x.get("value")]
    hits=retrieve(query if not context else query+" "+" ".join(context[:8]),top_k); warnings=[]; authorities=[]
    try:g=get_grounding()
    except IntegrationError as exc:g=None; warnings.append(f"Grounding unavailable: {exc}")
    for h in hits:
        meta=h.get("metadata",{}) if isinstance(h.get("metadata"),Mapping) else {}
        item={"case_name":meta.get("case_name") or h.get("case_name"),"court":meta.get("court") or h.get("court"),"date_year":meta.get("date_year") or meta.get("year") or h.get("year"),"citation":meta.get("citation") or h.get("citation"),"relevant_passage":h["text"],"source_url":meta.get("source_url") or h.get("source_url"),"source_type":meta.get("source_type") or h.get("source_type") or "UPLOADED_CASE_EVIDENCE","source":{"document":h.get("document"),"page":h.get("page"),"chunk_id":h.get("chunk_id")}}
        item["citation_status"]="UNVERIFIED"
        if g and item["citation"]:
            probe={"claim_text":item["relevant_passage"],**item["source"]}
            validated=g.validate_citations([probe],hits,[]); item["citation_status"]="VERIFIED" if validated and validated[0].get("valid") else "UNVERIFIED"
        if item["source_type"]=="EXTERNAL_LEGAL_AUTHORITY" and item["citation_status"]!="VERIFIED": warnings.append(f"Unverified external authority: {item['case_name'] or 'unnamed source'}.")
        authorities.append(item)
    return {"query":query,"authorities":authorities,"warnings":warnings,"case_context_used":context}
