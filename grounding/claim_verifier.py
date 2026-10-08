"""Conservative, evidence-first claim verification with contradiction checks."""
from __future__ import annotations
import re
from typing import Any
STATUSES={"SUPPORTED","PARTIALLY_SUPPORTED","UNSUPPORTED","CONTRADICTED","UNCERTAIN"}
_NEG={"not","never","no","none","neither","without","didn't","didnt","wasn't","wasnt","cannot","can't","cant"}
_STOP={"the","a","an","was","were","is","are","of","on","in","to","and","or","for","with","from","that","this","by","as","at"}
def _norm(text): return re.sub(r"[^a-z0-9]+"," ",str(text or "").lower()).strip()
def _tokens(text): return {t for t in _norm(text).split() if t not in _STOP and len(t)>1}
def _negated(text): return bool(_NEG & set(_norm(text).split()))
def _evidence_items(context:Any):
    if context is None:return []
    if isinstance(context,dict):
        for k in ("evidence","sources","context","retrieved"):
            if k in context:return _evidence_items(context[k])
        return [context] if context.get("text") else []
    if isinstance(context,(list,tuple)):return [x for x in context if isinstance(x,dict) and x.get("text")]
    return []
def _numbers(text): return set(re.findall(r"\b\d+(?:[.,]\d+)?\b",str(text or "")))
def _support_score(claim,evidence):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(evidence))
    if len(sentences) > 1:
        return max((_support_score(claim, sentence) for sentence in sentences), default=0.0)
    ct,et=_tokens(claim),_tokens(evidence)
    if not ct or not et:return 0.0
    cn,en=_numbers(claim),_numbers(evidence)
    if cn and cn!=en:return 0.0
    if _norm(claim)==_norm(evidence):return 1.0
    return len(ct&et)/len(ct)
def verify_claims(claims:list[dict],context:Any,min_support:float=.72)->list[dict]:
    evidence=_evidence_items(context); out=[]
    for original in claims or []:
        claim=dict(original); positives=[]; opposites=[]
        for item in evidence:
            for text in re.split(r"(?<=[.!?])\s+|\n+", item.get("text", "")):
                score = _support_score(claim.get("claim_text", ""), text)
                same_neg = _negated(claim.get("claim_text", "")) == _negated(text)
                if score >= min_support and same_neg:
                    positives.append((score, item))
                elif score >= min_support and not same_neg:
                    opposites.append((score, item))
        positives.sort(key=lambda x:x[0],reverse=True); opposites.sort(key=lambda x:x[0],reverse=True)
        if positives:
            score,best=positives[0]; claim["status"]="SUPPORTED" if score>=.85 else "PARTIALLY_SUPPORTED"
            claim["supporting_evidence"]=[dict(x[1]) for x in positives]
            claim.update(document=best.get("document"),page=best.get("page"),chunk_id=best.get("chunk_id"),confidence=round(min(.99,max(.5,score)),2))
        elif opposites:
            _,best=opposites[0]; claim["status"]="CONTRADICTED"; claim["supporting_evidence"]=[dict(best)]
            claim.update(document=best.get("document"),page=best.get("page"),chunk_id=best.get("chunk_id"),confidence=.8)
        else:
            claim["status"]="UNSUPPORTED" if evidence else "UNCERTAIN"; claim["supporting_evidence"]=[]; claim["confidence"]=0.0
        out.append(claim)
    return out
