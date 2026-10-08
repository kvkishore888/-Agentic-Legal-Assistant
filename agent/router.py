"""Deterministic routing for four legal workflows."""
from __future__ import annotations
import re
ROUTES=("CASE_CONTRACT_REVIEW","LEGAL_DRAFTING","LEGAL_RESEARCH","GROUNDED_RAG_CHAT")
_PATTERNS={"CASE_CONTRACT_REVIEW":r"\b(review|analyse|analyze|contract|agreement|clause|case file|case review|document review)\b","LEGAL_DRAFTING":r"\b(draft|drafting|write|prepare|compose|template|petition|notice|agreement|clause)\b","LEGAL_RESEARCH":r"\b(research|precedent|judgment|statute|section|law|legal position|authorit)\b"}
def route(query:str)->str:
    q=str(query or "").lower(); scores={name:len(re.findall(pattern,q)) for name,pattern in _PATTERNS.items()}; best=max(scores,key=scores.get)
    if scores[best]==0:return "GROUNDED_RAG_CHAT"
    if scores.get("CASE_CONTRACT_REVIEW",0)>0 and re.search(r"\b(review|analyse|analyze)\b",q):return "CASE_CONTRACT_REVIEW"
    return best
def route_request(query:str)->dict:return {"route":route(query),"workflow":route(query).lower(),"query":query}
