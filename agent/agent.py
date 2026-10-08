"""Conversational, evidence-grounded legal agent.

The LLM is used for natural-language reasoning and conversation. It is never
treated as a source of truth: retrieved evidence is passed separately, and
the existing deterministic claim/citation verification remains the final gate.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from retrieval.hybrid_search import retrieve
from grounding.claim_extractor import extract_claims
from grounding.claim_verifier import verify_claims
from grounding.citation_validator import validate_citations
from .router import route_request

_UNVERIFIED = "Not verified from the provided sources."


def _evidence(context: Any) -> list[dict[str, Any]]:
    if isinstance(context, dict):
        for key in ("evidence", "sources", "context", "retrieved"):
            if key in context:
                return _evidence(context[key])
    if isinstance(context, (list, tuple)):
        return [
            item for item in context
            if isinstance(item, dict) and item.get("text")
        ]
    return []


def _format_evidence(evidence: list[dict[str, Any]]) -> str:
    blocks = []
    for i, item in enumerate(evidence, 1):
        blocks.append(
            f"[EVIDENCE {i}]\n"
            f"Document: {item.get('document', 'Unknown')}\n"
            f"Page: {item.get('page', 'Unknown')}\n"
            f"Chunk: {item.get('chunk_id', 'Unknown')}\n"
            f"Text: {item.get('text', '')}"
        )
    return "\n\n".join(blocks)


def _history_messages(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    messages = []
    for item in (history or [])[-8:]:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def _llm_candidate(
    query: str,
    evidence: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
) -> str | None:
    """Generate a conversational candidate answer, or None when LLM is unavailable."""
    provider = os.getenv("LLM_PROVIDER", "none").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if provider != "openai" or not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = os.getenv("LLM_MODEL", "gpt-6-luna")

        instructions = """You are an evidence-grounded legal assistant.

Rules:
1. Retrieved evidence is the only source of factual/legal claims about the user's
   documents. Do not invent facts, dates, names, sections, cases, citations,
   holdings, or procedural events.
2. Conversation history is context for resolving follow-up questions, NOT evidence.
3. If the evidence does not establish an answer, explicitly say that it is not
   verified from the provided sources.
4. Explain the answer naturally and conversationally. For follow-up questions,
   resolve references such as "that case", "the second document", or "when was it?"
   from the conversation context.
5. Prefer precise source references in the prose, using the supplied document,
   page, and chunk identifiers.
6. Do not claim that external legal research was performed unless the supplied
   evidence explicitly identifies an independently verified external authority.
7. This is legal information support, not a substitute for a qualified lawyer.

Return only the answer text; do not output JSON or analysis."""

        history = _history_messages(conversation_history)
        prompt = (
            "RETRIEVED EVIDENCE:\n"
            f"{_format_evidence(evidence) or '[NO EVIDENCE RETRIEVED]'}\n\n"
            "USER QUESTION:\n"
            f"{query}"
        )

        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=history + [{"role": "user", "content": prompt}],
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "700")),
        )
        answer = (response.output_text or "").strip()
        return answer or None
    except Exception as exc:
        # A provider failure must never bypass grounding. The deterministic
        # fallback remains safe. Do not expose provider error details to users.
        _ = exc
        return None


def _candidate(
    query: str,
    context: Any,
    generator: Callable[[str, Any], str] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[str, bool]:
    evidence = _evidence(context)
    if generator:
        return str(generator(query, context) or ""), False

    llm_answer = _llm_candidate(query, evidence, conversation_history)
    if llm_answer:
        return llm_answer, True

    if isinstance(context, dict) and context.get("candidate_answer"):
        return str(context["candidate_answer"]), False
    if isinstance(context, dict) and context.get("answer"):
        return str(context["answer"]), False
    return _extractive_fallback(query, evidence), False

def _extractive_fallback(query: str, evidence: list[dict[str, Any]]) -> str:
    """Return a small, grounded answer from the best matching evidence sentences."""
    import re

    stop = {"what","when","where","who","which","how","why","was","were","is","are","the","a","an","of","on","in","to","for","from","did","does","do","tell","me","about","and","or","with"}
    query_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2 and t not in stop}
    candidates = []
    for item in evidence:
        text = str(item.get("text", ""))
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(sentence.split()) < 4:
                continue
            low = sentence.lower()
            if "testing notes for the chatbot" in low or "ask the chatbot questions" in low:
                continue
            words = set(re.findall(r"[a-z0-9]+", low))
            overlap = len(query_tokens & words)
            if overlap:
                score = overlap / max(1, len(query_tokens))
                if any(ch.isdigit() for ch in sentence):
                    score += 0.08
                candidates.append((score, -len(sentence), sentence))
    candidates.sort(reverse=True)
    if not candidates:
        return _UNVERIFIED
    return candidates[0][2]


def _confidence(
    claims: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> str:
    if not claims or not evidence:
        return "LOW"
    supported = sum(c.get("status") == "SUPPORTED" for c in claims)
    contradicted = sum(c.get("status") == "CONTRADICTED" for c in claims)
    valid = sum(bool(c.get("valid")) for c in citations)
    if contradicted:
        return "LOW"
    ratio = supported / len(claims)
    if ratio == 1 and (not citations or valid == len(citations)):
        return "HIGH"
    if ratio >= 0.5:
        return "MEDIUM"
    return "LOW"


def answer_with_grounding(
    query: str,
    context: Any = None,
    candidate_generator: Callable[[str, Any], str] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    evidence = _evidence(context)
    candidate, llm_used = _candidate(
        query, context, candidate_generator, conversation_history
    )

    claims = verify_claims(extract_claims(candidate), evidence)
    citations = []
    for claim in claims:
        if (
            claim.get("status") == "SUPPORTED"
            and all(claim.get(key) is not None for key in ("document", "page", "chunk_id"))
        ):
            citations.append(
                {
                    "claim_text": claim["claim_text"],
                    "document": claim["document"],
                    "page": claim["page"],
                    "chunk_id": claim["chunk_id"],
                }
            )

    citations = validate_citations(citations, evidence, claims)
    established = [
        claim["claim_text"]
        for claim in claims
        if claim.get("status") == "SUPPORTED"
    ]

    # Preserve the conversational LLM response only when every extracted claim
    # is supported and every supported claim has source metadata. Otherwise,
    # fall back to verified atomic claims so the model cannot introduce an
    # unsupported legal fact.
    all_supported = bool(claims) and all(
        claim.get("status") == "SUPPORTED" for claim in claims
    )
    all_cited = bool(claims) and all(
        claim.get("status") == "SUPPORTED"
        and claim.get("document") is not None
        and claim.get("page") is not None
        and claim.get("chunk_id") is not None
        for claim in claims
    )
    if all_supported and all_cited and candidate.strip():
        answer = candidate.strip()
    else:
        answer = " ".join(established) if established else _UNVERIFIED
    unsupported = [
        claim for claim in claims if claim.get("status") != "SUPPORTED"
    ]

    return {
        "answer": answer,
        "candidate_answer": candidate,
        "claims": claims,
        "citations": citations,
        "unsupported_claims": unsupported,
        "confidence": _confidence(claims, citations, evidence),
        "missing_information": (
            [] if established
            else ["supporting evidence for the requested factual claims"]
        ),
        "route": route_request(query)["route"],
        # This indicates configuration, not a successful provider call. The UI
        # should distinguish enabled configuration from an actual LLM response.
        "llm_enabled": os.getenv("LLM_PROVIDER", "none").lower() == "openai"
        and bool(os.getenv("OPENAI_API_KEY")),
        "llm_used": llm_used,
    }


def grounded_rag_chat(
    query: str,
    top_k: int = 5,
    candidate_generator=None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    evidence = retrieve(query, top_k=top_k)
    return answer_with_grounding(
        query,
        evidence,
        candidate_generator=candidate_generator,
        conversation_history=conversation_history,
    )
