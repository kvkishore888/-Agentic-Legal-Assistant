"""Conservative evidence-based claim verification."""
from __future__ import annotations
import re
from typing import Any
STATUSES={"SUPPORTED","PARTIALLY_SUPPORTED","UNSUPPORTED","CONTRADICTED","UNCERTAIN"}
_NEG={"not","never","no","none","neither","without","didn't","didnt","wasn't","wasnt","cannot","can't","cant"}
_STOP={"the","a","an","was","were","is","are","of","on","in","to","and","or","for","with","from","that","this","by","as","at"}
def _norm(text): return re.sub(r"[^a-z0-9]+"," ",str(text or "").lower()).strip()
def _tokens(text): return {t for t in _norm(text).split() if t not in _STOP and len(t)>1}
def _negated(text): return any(w in _NEG for w in _norm(text).split())
def _evidence_items(context: Any):
    if context is None: return []
    if isinstance(context,dict):
        for k in ("evidence","sources","context","retrieved"):
            if k in context: return _evidence_items(context[k])
        return [context] if context.get("text") else []
    if isinstance(context,(list,tuple)): return [x for x in context if isinstance(x,dict) and x.get("text")]
    return []
def _support_score(claim,evidence):
    ct,et=_tokens(claim),_tokens(evidence)
    if not ct or not et: return 0.0
    if _norm(claim)==_norm(evidence): return 1.0
    return len(ct&et)/len(ct)
def verify_claims(claims:list[dict],context:Any,min_support:float=0.72)->list[dict]:
    evidence=_evidence_items(context); verified=[]
    for original in claims or []:
        claim=dict(original); matches=[]; contradictions=[]
        for item in evidence:
            score=_support_score(claim.get("claim_text",""),item.get("text",""))
            if score>=min_support: matches.append((score,item))
            elif score>=0.72 and _negated(item.get("text","")) != _negated(claim.get("claim_text","")): contradictions.append((score,item))
        matches.sort(key=lambda x:x[0],reverse=True)
        if matches:
            score,best=matches[0]; claim["status"]="SUPPORTED" if score>=0.85 else "PARTIALLY_SUPPORTED"
            claim["supporting_evidence"]=[dict(x[1]) for x in matches]; claim["document"]=best.get("document"); claim["page"]=best.get("page"); claim["chunk_id"]=best.get("chunk_id"); claim["confidence"]=round(min(.99,max(.5,score)),2)
        elif contradictions:
            _,best=contradictions[0]; claim["status"]="CONTRADICTED"; claim["supporting_evidence"]=[dict(best)]; claim["document"]=best.get("document"); claim["page"]=best.get("page"); claim["chunk_id"]=best.get("chunk_id"); claim["confidence"]=.8
        else:
            claim["status"]="UNSUPPORTED" if evidence else "UNCERTAIN"; claim["supporting_evidence"]=[]; claim["confidence"]=0.0
        if claim["status"] not in STATUSES: claim["status"]="UNCERTAIN"
        verified.append(claim)
    return verified
