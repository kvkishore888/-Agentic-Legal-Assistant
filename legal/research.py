"""Grounded legal research over indexed statutes, judgments, and case documents."""
from __future__ import annotations
from typing import Mapping, Any
from .shared import retrieve, get_grounding, IntegrationError

def research(query: str, case: Mapping[str, Any] | None = None, *, top_k: int = 5) -> dict[str, Any]:
    context = [str(x.get("value")) for x in (case or {}).get("key_facts", []) if x.get("value")]
    hits = retrieve(query if not context else query + " " + " ".join(context[:8]), top_k)
    warnings = []
    authorities = []
    try:
        grounding = get_grounding()
    except IntegrationError as exc:
        grounding = None
        warnings.append(f"Grounding unavailable: {exc}")
    for h in hits:
        meta = h.get("metadata", {}) if isinstance(h.get("metadata"), Mapping) else {}
        source_type = meta.get("source_type") or h.get("source_type") or "UPLOADED_CASE_EVIDENCE"
        item = {
            "case_name": meta.get("case_name") or h.get("case_name"),
            "court": meta.get("court") or h.get("court"),
            "date_year": meta.get("date_year") or meta.get("year") or h.get("year"),
            "citation": meta.get("citation") or h.get("citation"),
            "relevant_passage": h["text"],
            "source_url": meta.get("source_url") or h.get("source_url"),
            "source_type": source_type,
            "source": {"document": h.get("document"), "page": h.get("page"), "chunk_id": h.get("chunk_id")},
            "citation_status": "UNVERIFIED",
        }
        if grounding and item["citation"]:
            probe = {"claim_text": item["relevant_passage"], **item["source"]}
            checked = grounding.validate_citations([probe], hits, [])
            item["citation_status"] = "VERIFIED" if checked and checked[0].get("valid") else "UNVERIFIED"
        if source_type == "EXTERNAL_LEGAL_AUTHORITY" and item["citation_status"] != "VERIFIED":
            warnings.append(f"Unverified external authority: {item['case_name'] or 'unnamed source'}.")
        if source_type != "EXTERNAL_LEGAL_AUTHORITY":
            item["authority_scope"] = "uploaded evidence; not independently verified as an external authority"
        authorities.append(item)
    if not any(a.get("source_type") == "EXTERNAL_LEGAL_AUTHORITY" for a in authorities):
        warnings.append("No independently verified external legal authority was retrieved. Upload or index authoritative statutes/judgments before treating research as external legal research.")
    return {"query": query, "authorities": authorities, "warnings": warnings, "case_context_used": context}
