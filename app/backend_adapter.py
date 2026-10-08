"""Application adapter: one stable UI API over Members 1–3."""
from __future__ import annotations
from typing import Any
from agent.agent import answer_with_grounding
from agent.router import route_request
from legal.case_review import review_case
from legal.drafting import draft_document
from legal.research import research
from retrieval.hybrid_search import retrieve as shared_retrieve

DRAFT_TYPES = ("legal notice", "petition", "affidavit", "bail")

def retrieve(query: str, top_k: int = 5):
    return list(shared_retrieve(query, top_k=top_k))

def run_workflow(workflow: str, query: str, top_k: int = 5, document_type: str = "legal notice") -> dict[str, Any]:
    evidence = retrieve(query, top_k)
    if workflow == "Case / Contract Review":
        raw = review_case(query, top_k=top_k)
        return {
            "answer": "Case/contract review completed from retrieved evidence.",
            "route": "CASE_CONTRACT_REVIEW",
            "evidence": evidence,
            **raw,
        }
    if workflow == "Legal Drafting":
        if document_type not in DRAFT_TYPES:
            raise ValueError(f"Unsupported document type: {document_type}")
        raw = draft_document(document_type, query, top_k=top_k)
        return {"route": "LEGAL_DRAFTING", "evidence": evidence, **raw}
    if workflow == "Legal Research":
        raw = research(query, top_k=top_k)
        return {
            "answer": "Research results are shown below. External authorities remain unverified unless their citation is validated.",
            "route": "LEGAL_RESEARCH",
            "evidence": evidence,
            **raw,
        }
    raw = answer_with_grounding(query, evidence)
    return {"route": "GROUNDED_RAG_CHAT", "evidence": evidence, **raw}

def route(query: str):
    return route_request(query)
