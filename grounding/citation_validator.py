"""Citation validation against retrieved evidence only."""
from __future__ import annotations
from typing import Any
def _items(context:Any):
    if isinstance(context,dict):
        for k in ("evidence","sources","context","retrieved"):
            if k in context:return _items(context[k])
        return [context] if context.get("text") else []
    if isinstance(context,(list,tuple)): return [x for x in context if isinstance(x,dict)]
    return []
def validate_citations(citations:list[dict]|None,context:Any,claims:list[dict]|None=None)->list[dict]:
    evidence=_items(context); by_chunk={str(x.get("chunk_id")):x for x in evidence if x.get("chunk_id") is not None}; by_doc_page={(str(x.get("document")),str(x.get("page"))):x for x in evidence if x.get("document") is not None and x.get("page") is not None}; claim_by_chunk={str(c.get("chunk_id")):c for c in (claims or []) if c.get("chunk_id") is not None}; out=[]
    for citation in citations or []:
        c=dict(citation); chunk=str(c.get("chunk_id")) if c.get("chunk_id") is not None else None; doc=c.get("document"); page=c.get("page"); source=by_chunk.get(chunk) if chunk else by_doc_page.get((str(doc),str(page)))
        checks={"source_exists":bool(evidence),"document_exists":False,"page_exists":False,"chunk_exists":False,"evidence_supports_claim":False,"relevant":False}
        if source:
            checks["document_exists"]=doc is None or str(source.get("document"))==str(doc); checks["page_exists"]=page is None or str(source.get("page"))==str(page); checks["chunk_exists"]=chunk is None or str(source.get("chunk_id"))==chunk
            checks["evidence_supports_claim"]=bool(c.get("claim_text")) and c["claim_text"].lower() in str(source.get("text","")).lower(); linked=claim_by_chunk.get(chunk) if chunk else None; checks["relevant"]=bool(linked and linked.get("status")=="SUPPORTED") or checks["evidence_supports_claim"]
        c["valid"]=all(checks.values()); c["checks"]=checks
        if source:c["evidence"]=dict(source)
        out.append(c)
    return out
