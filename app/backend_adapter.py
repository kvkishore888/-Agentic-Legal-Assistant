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

def run_workflow(workflow: str, query: str, top_k: int = 5, document_type: str = "legal notice",
                 conversation_context: list[dict[str, str]] | None = None) -> dict[str, Any]:
    context = conversation_context or []
    # Retrieval must be driven by the CURRENT user question. Conversation
    # history is passed separately to the conversational agent so follow-ups
    # can be resolved without polluting evidence retrieval with old questions.
    retrieval_query = query.strip()
    if not retrieval_query:
        raise ValueError("Please enter a question.")
    evidence = retrieve(retrieval_query, top_k)
    if workflow == "Case / Contract Review":
        raw = review_case(query, top_k=top_k)
        return {"answer": "Case/contract review completed from retrieved evidence.",
                "route": "CASE_CONTRACT_REVIEW", "evidence": evidence, **raw}
    if workflow == "Legal Drafting":
        if document_type not in DRAFT_TYPES:
            raise ValueError(f"Unsupported document type: {document_type}")
        raw = draft_document(document_type, query, top_k=top_k)
        return {"route": "LEGAL_DRAFTING", "evidence": evidence, **raw}
    if workflow == "Legal Research":
        raw = research(query, top_k=top_k)
        return {"answer": "Research is grounded in indexed sources. External authority is shown as verified only when its source metadata and evidence validate it.",
                "route": "LEGAL_RESEARCH", "evidence": evidence, **raw}
    raw = answer_with_grounding(query, evidence, conversation_history=context)
    # The selected UI workflow is authoritative; the natural-language router
    # is useful for intent hints but must not relabel the active workflow.
    return {"route": "GROUNDED_RAG_CHAT", "evidence": evidence, **raw}

def route(query: str):
    return route_request(query)
