"""Strict citation validation: identity, source existence, and verified claim linkage."""
from __future__ import annotations
from typing import Any

def _items(context: Any):
    if isinstance(context, dict):
        for k in ("evidence", "sources", "context", "retrieved"):
            if k in context:
                return _items(context[k])
        return [context] if context.get("text") else []
    if isinstance(context, (list, tuple)):
        return [x for x in context if isinstance(x, dict) and x.get("text")]
    return []

def validate_citations(citations: list[dict] | None, context: Any, claims: list[dict] | None = None) -> list[dict]:
    evidence = _items(context)
    by_chunk = {str(x.get("chunk_id")): x for x in evidence if x.get("chunk_id") is not None}
    by_doc_page = {
        (str(x.get("document")), str(x.get("page"))): x
        for x in evidence
        if x.get("document") is not None and x.get("page") is not None
    }
    def claim_key(item):
        return (str(item.get("document")), str(item.get("page")),
                str(item.get("chunk_id")), str(item.get("claim_text", "")).strip())
    claim_by_key = {claim_key(claim): claim for claim in claims or []}

    out = []
    for citation in citations or []:
        c = dict(citation)
        chunk = str(c.get("chunk_id")) if c.get("chunk_id") is not None else None
        doc, page = c.get("document"), c.get("page")
        source = by_chunk.get(chunk) if chunk else by_doc_page.get((str(doc), str(page)))
        linked = claim_by_key.get(claim_key(c))
        checks = {
            "source_exists": bool(source),
            "document_exists": bool(source) and (doc is None or str(source.get("document")) == str(doc)),
            "page_exists": bool(source) and (page is None or str(source.get("page")) == str(page)),
            "chunk_exists": bool(source) and (chunk is None or str(source.get("chunk_id")) == chunk),
            # A citation is valid only when the verifier itself marked the linked
            # claim SUPPORTED. We intentionally do not use substring matching:
            # a citation cannot turn an unsupported claim into a valid one.
            "claim_verified": bool(linked and linked.get("status") == "SUPPORTED"),
            "relevant": bool(linked and linked.get("status") == "SUPPORTED"),
        }
        c["valid"] = all(checks.values())
        c["checks"] = checks
        if source:
            c["evidence"] = dict(source)
        out.append(c)
    return out
