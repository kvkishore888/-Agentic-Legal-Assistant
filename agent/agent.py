"""Grounded orchestration: retrieve -> candidate -> verify -> final answer."""
from __future__ import annotations
from typing import Any,Callable
from retrieval.hybrid_search import retrieve
from grounding.claim_extractor import extract_claims
from grounding.claim_verifier import verify_claims
from grounding.citation_validator import validate_citations
from .router import route_request
_UNVERIFIED="Not verified from the provided sources."
def _evidence(context:Any):
    if isinstance(context,dict):
        for k in ("evidence","sources","context","retrieved"):
            if k in context:return _evidence(context[k])
    if isinstance(context,(list,tuple)):return [x for x in context if isinstance(x,dict) and x.get("text")]
    return []
def _candidate(query,context,generator):
    if generator:return str(generator(query,context) or "")
    if isinstance(context,dict) and context.get("candidate_answer"):return str(context["candidate_answer"])
    if isinstance(context,dict) and context.get("answer"):return str(context["answer"])
    return "\n".join(str(x["text"]) for x in _evidence(context))
def _confidence(claims,citations,evidence):
    if not claims or not evidence:return "LOW"
    supported=sum(c.get("status")=="SUPPORTED" for c in claims); contradicted=sum(c.get("status")=="CONTRADICTED" for c in claims); valid=sum(c.get("valid") for c in citations)
    if contradicted:return "LOW"
    ratio=supported/len(claims)
    if ratio==1 and (not citations or valid==len(citations)):return "HIGH"
    if ratio>=.5:return "MEDIUM"
    return "LOW"
def answer_with_grounding(query:str,context:Any=None,candidate_generator:Callable[[str,Any],str]|None=None)->dict:
    evidence=_evidence(context); candidate=_candidate(query,context,candidate_generator); claims=verify_claims(extract_claims(candidate),evidence); citations=[]
    for c in claims:
        if c.get("status")=="SUPPORTED" and all(c.get(k) is not None for k in ("document","page","chunk_id")):citations.append({"claim_text":c["claim_text"],"document":c["document"],"page":c["page"],"chunk_id":c["chunk_id"]})
    citations=validate_citations(citations,evidence,claims); established=[c["claim_text"] for c in claims if c.get("status")=="SUPPORTED"]; answer=" ".join(established) if established else _UNVERIFIED; unsupported=[c for c in claims if c.get("status")!="SUPPORTED"]
    return {"answer":answer,"claims":claims,"citations":citations,"unsupported_claims":unsupported,"confidence":_confidence(claims,citations,evidence),"missing_information":[] if established else ["supporting evidence for the requested factual claims"],"route":route_request(query)["route"]}
def grounded_rag_chat(query:str,top_k:int=5,candidate_generator=None)->dict:
    return answer_with_grounding(query,retrieve(query,top_k=top_k),candidate_generator=candidate_generator)
