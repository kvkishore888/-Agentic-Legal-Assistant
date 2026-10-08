"""Grounding, retrieval, citation and usefulness metrics; no fabricated results."""
from __future__ import annotations
import re

def _norm(t):
    return set(re.findall(r"[a-z0-9]+", str(t or "").lower()))

def retrieval_recall_at_k(results, expected, k=5):
    exp = {(x.get("document"), x.get("page"), x.get("chunk_id")) for x in expected}
    if not exp:
        return 0.0
    got = {(x.get("document"), x.get("page"), x.get("chunk_id")) for x in results[:k]}
    return len(exp & got) / len(exp)

def citation_correctness(citations, expected):
    exp = {(x.get("document"), x.get("page"), x.get("chunk_id")) for x in expected}
    if not citations:
        return 0.0
    return sum((x.get("document"), x.get("page"), x.get("chunk_id")) in exp for x in citations) / len(citations)

def groundedness(claims, evidence):
    if not claims:
        return 0.0
    evidence_tokens = _norm(" ".join(x.get("text", "") for x in evidence))
    good = 0
    for claim in claims:
        if isinstance(claim, str):
            text, status = claim, ""
        else:
            text = claim.get("claim_text") or claim.get("claim") or claim.get("text", "")
            status = str(claim.get("status", "")).upper()
        if status:
            good += status == "SUPPORTED"
        else:
            tokens = _norm(text)
            good += bool(tokens) and tokens <= evidence_tokens
    return good / len(claims)

def fabricated_claim_rate(claims, evidence):
    return 1 - groundedness(claims, evidence) if claims else 0.0

def answer_usefulness(answer, expected):
    e, a = _norm(expected), _norm(answer)
    return len(e & a) / len(e) if e else 0.0

def evaluate_result(case, result):
    evidence = result.get("evidence") or []
    claims = result.get("claims") or []
    citations = result.get("citations") or []
    return {
        "id": case["id"],
        "groundedness": groundedness(claims, evidence),
        "retrieval_quality": retrieval_recall_at_k(evidence, case.get("expected_sources", [])),
        "citation_correctness": citation_correctness(citations, case.get("expected_sources", [])),
        "fabricated_claim_rate": fabricated_claim_rate(claims, evidence),
        "answer_usefulness": answer_usefulness(result.get("answer", ""), case.get("expected_answer", "")),
    }

def aggregate(rows):
    if not rows:
        return {}
    keys = ("groundedness", "retrieval_quality", "citation_correctness", "fabricated_claim_rate", "answer_usefulness")
    return {k: sum(float(r.get(k, 0)) for r in rows) / len(rows) for k in keys}
