"""Evidence-first legal drafting with explicit placeholders and verification gates."""
from __future__ import annotations
from .shared import retrieve, source_ref, get_grounding, IntegrationError
from .missing_info import detect_missing_information, requirements_for
from grounding.claim_extractor import extract_claims

_SECTIONS = {
    "bail": ("BAIL APPLICATION", ["IN THE APPROPRIATE COURT", "CASE DETAILS", "FACTS SUPPORTED BY THE RECORD", "GROUNDS", "PRAYER"]),
    "petition": ("PETITION", ["IN THE APPROPRIATE COURT", "PARTIES", "MATERIAL FACTS", "GROUNDS", "RELIEF / PRAYER"]),
    "legal notice": ("LEGAL NOTICE", ["SENDER", "RECIPIENT", "BACKGROUND", "BREACH / GRIEVANCE", "DEMAND / RELIEF"]),
    "affidavit": ("AFFIDAVIT", ["DEPONENT", "FACTS", "VERIFICATION"]),
}

def _draft_text(document_type: str, hits: list[dict], missing: list[dict]) -> str:
    title, sections = _SECTIONS[document_type.lower()]
    lines = [f"# {title}", "", "## Grounding status",
             "This draft contains only facts present in the retrieved documents. Missing material is explicitly marked as a placeholder.",
             ""]
    for section in sections:
        lines += [f"## {section}"]
        if section in {"FACTS SUPPORTED BY THE RECORD", "MATERIAL FACTS", "FACTS", "BACKGROUND", "BREACH / GRIEVANCE"}:
            if hits:
                lines += [f"- {h['text']} [Source: {h['document']}, p. {h['page']}, {h['chunk_id']}]" for h in hits]
            else:
                lines.append("- [MISSING — supporting source material]")
        else:
            lines.append("- [MISSING — provide verified information before filing]")
        lines.append("")
    if missing:
        lines += ["## Missing information"]
        lines += [f"- [MISSING — {m['item']}] {m['message']}" for m in missing]
    lines += ["", "## Verification warning",
              "This is an evidence-grounded working draft, not a filing-ready legal document. Review every placeholder and citation before use."]
    return "\n".join(lines)

def draft_document(document_type: str, query: str, *, top_k: int = 5, required_information: list[str] | None = None) -> dict:
    key = document_type.strip().lower()
    if key not in _SECTIONS:
        raise ValueError(f"Unsupported document type: {document_type}")
    required = required_information or requirements_for(key)
    hits = retrieve(query, top_k)
    missing = detect_missing_information(hits, required)
    candidate = " ".join(h["text"] for h in hits)
    verification = []
    citations = []
    try:
        g = get_grounding()
        verification = g.verify_claims(extract_claims(candidate), hits)
        for c in verification:
            if c.get("status") == "SUPPORTED" and c.get("chunk_id") is not None:
                citations.append({"claim_text": c.get("claim_text"), "document": c.get("document"),
                                  "page": c.get("page"), "chunk_id": c.get("chunk_id")})
        citations = g.validate_citations(citations, hits, verification)
    except IntegrationError as exc:
        verification = [{"status": "VERIFICATION_UNAVAILABLE", "message": str(exc)}]
    safe = bool(verification) and all("claim_text" in c and c.get("status") == "SUPPORTED" for c in verification) and bool(citations) and all(c.get("valid") for c in citations)
    status = "READY_FOR_REVIEW" if safe and not missing else "INCOMPLETE"
    return {
        "document_type": key,
        "status": status,
        "draft": _draft_text(key, hits, missing),
        "missing_information": missing,
        "claim_verification": verification,
        "citations": citations,
        "sources": [source_ref(h) for h in hits],
        "warnings": ["No facts are added beyond retrieved evidence.",
                     "Filing-ready legal review is still required; placeholders must be resolved."] if missing else
                    ["Draft is evidence-grounded but requires legal review before filing."]
    }
