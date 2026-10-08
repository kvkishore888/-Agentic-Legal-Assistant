"""PDF ingestion with page preservation and automatic OCR fallback."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .metadata import build_metadata
from .ocr import needs_ocr, ocr_page

def load_pdf(path: str, use_ocr: bool = True, ocr_language: str = "eng", ocr_min_chars: int = 40) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF ingestion") from exc
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(path)
    pages = []
    with fitz.open(str(pdf_path)) as document:
        for index, page in enumerate(document):
            text = page.get_text("text") or ""
            ocr_used = False
            if use_ocr and needs_ocr(text, ocr_min_chars):
                try:
                    ocr_text = ocr_page(page, language=ocr_language)
                    if len(ocr_text) > len(text.strip()):
                        text, ocr_used = ocr_text, True
                except RuntimeError:
                    pass
            pages.append({"text": text, "document": pdf_path.name, "page": index + 1, "section": "", "metadata": build_metadata(pdf_path.name, index + 1, source_path=str(pdf_path), ocr_used=ocr_used)})
    return pages

def load_documents(paths: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    pages = []
    for path in paths:
        pages.extend(load_pdf(path, **kwargs))
    return pages
