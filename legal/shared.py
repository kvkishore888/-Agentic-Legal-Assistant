"""Shared adapters for retrieval and the Member 2 grounding interfaces."""
from __future__ import annotations
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping
RetrieveFn=Callable[...,Iterable[Mapping[str,Any]]]
class IntegrationError(RuntimeError): pass
def get_retriever()->RetrieveFn:
    try: module=importlib.import_module("retrieval.hybrid_search")
    except ImportError as exc: raise IntegrationError("Member 1 retrieval is unavailable") from exc
    fn=getattr(module,"retrieve",None)
    if not callable(fn): raise IntegrationError("retrieval.hybrid_search.retrieve is not configured")
    return fn
@dataclass(frozen=True)
class GroundingService:
    verify_claims: Callable[...,list[dict]]
    validate_citations: Callable[...,list[dict]]
_grounding:GroundingService|None=None
def configure_grounding(service:GroundingService)->None:
    global _grounding; _grounding=service
def get_grounding()->GroundingService:
    if _grounding:return _grounding
    try:
        verifier=importlib.import_module("grounding.claim_verifier")
        validator=importlib.import_module("grounding.citation_validator")
    except ImportError as exc: raise IntegrationError("Member 2 grounding is unavailable") from exc
    verify=getattr(verifier,"verify_claims",None); validate=getattr(validator,"validate_citations",None)
    if not callable(verify) or not callable(validate): raise IntegrationError("Member 2 grounding APIs are unavailable")
    return GroundingService(verify,validate)
def normalize_result(item:Mapping[str,Any])->dict[str,Any]:
    if not isinstance(item,Mapping): raise ValueError("retrieval result must be a mapping")
    text=str(item.get("text",""))
    if not text.strip(): raise ValueError("retrieval result has no text")
    result=dict(item); result["document"]=str(item.get("document","UNKNOWN")); result["page"]=item.get("page","UNKNOWN"); result["chunk_id"]=item.get("chunk_id","UNKNOWN"); result["text"]=text
    if "score" in item:
        try: result["score"]=float(item["score"])
        except (TypeError,ValueError): pass
    result["source"]=f'{result["document"]} — Page {result["page"]}'
    return result
def retrieve(query:str,top_k:int=5)->list[dict[str,Any]]:
    if not query or not query.strip(): raise ValueError("query must not be empty")
    if top_k<1: raise ValueError("top_k must be >= 1")
    return [normalize_result(x) for x in get_retriever()(query,top_k=top_k)]
def source_ref(hit:Mapping[str,Any])->dict[str,Any]:
    return {"document":hit.get("document","UNKNOWN"),"page":hit.get("page","UNKNOWN"),"chunk_id":hit.get("chunk_id","UNKNOWN"),"source":hit.get("source","UNKNOWN"),"text":hit.get("text","")}
