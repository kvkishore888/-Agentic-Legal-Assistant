"""Conservative contradiction detection with evidence from both sides."""
from __future__ import annotations
import re
from typing import Any,Mapping,Sequence
CONTRADICTION="CONTRADICTION"; POTENTIAL_CONTRADICTION="POTENTIAL_CONTRADICTION"; NO_CONTRADICTION="NO_CONTRADICTION"
_PATTERNS=(("date",r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s+\d{4})\b"),("amount",r"(?:₹|Rs\.?|INR|\$|USD)\s?[\d,]+(?:\.\d+)?"),("section",r"\b(?:Section|Sec\.?)\s*\d+[A-Za-z0-9()/-]*"))
def _value(s:str):
    for kind,pat in _PATTERNS:
        m=re.search(pat,s,re.I)
        if m:return kind,m.group(0).lower()
    return None
def compare_statements(left:Mapping[str,Any],right:Mapping[str,Any])->dict[str,Any]:
    a,b=str(left.get("text","")),str(right.get("text","")); va,vb=_value(a),_value(b)
    ta=set(re.findall(r"[a-z]{4,}",a.lower())); tb=set(re.findall(r"[a-z]{4,}",b.lower())); same=len(ta&tb)>=2
    status=NO_CONTRADICTION; reason="No sufficiently supported conflict detected."
    if same and va and vb and va[0]==vb[0] and va[1]!=vb[1]: status=CONTRADICTION; reason=f"Same topic has different {va[0]} values."
    elif same and va and vb and va[1]!=vb[1]: status=POTENTIAL_CONTRADICTION; reason="Related statements have differing extracted values; wording is insufficient for a direct contradiction."
    return {"status":status,"reason":reason,"statement_a":dict(left),"statement_b":dict(right)}
def find_contradictions(evidence:Sequence[Mapping[str,Any]])->list[dict[str,Any]]:
    return [r for i,a in enumerate(evidence) for b in evidence[i+1:] if (r:=compare_statements(a,b))["status"]!=NO_CONTRADICTION]
