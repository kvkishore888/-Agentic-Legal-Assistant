"""Legal-aware chunking that preserves page and source metadata."""
from .metadata import make_chunk_id, build_metadata

def _boundary(text, start, end):
    if end >= len(text):
        return end
    point = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind(": ", start, end))
    return point + 1 if point > start + (end - start) // 2 else end

def chunk_pages(pages, chunk_size: int = 1200, overlap: int = 180):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk parameters")
    out = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue
        document, page_number = page["document"], int(page["page"])
        metadata = dict(page.get("metadata") or {})
        metadata.update({"document": document, "page": page_number, "section": page.get("section", "") or ""})
        start, index = 0, 1
        while start < len(text):
            end = _boundary(text, start, min(len(text), start + chunk_size))
            chunk = text[start:end].strip()
            if chunk:
                chunk_id = make_chunk_id(document, page_number, index)
                chunk_metadata = dict(metadata)
                chunk_metadata["chunk_id"] = chunk_id
                # Keep evaluation/test instructions out of the legal evidence
                # corpus. This is intentionally conservative: a chunk is marked
                # instructional only when multiple explicit dataset markers occur.
                low_chunk = chunk.lower()
                markers = (
                    "test questions",
                    "grounding test rule",
                    "testing notes for the chatbot",
                    "ask the chatbot questions",
                )
                marker_count = sum(marker in low_chunk for marker in markers)
                chunk_metadata["content_type"] = (
                    "EVALUATION_INSTRUCTIONS" if marker_count >= 2 else "LEGAL_EVIDENCE"
                )
                out.append({"text": chunk, "document": document, "page": page_number, "section": page.get("section", "") or "", "chunk_id": chunk_id, "metadata": chunk_metadata})
                index += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return out
