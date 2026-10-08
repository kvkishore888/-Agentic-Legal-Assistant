"""PDF extraction with page-preserving metadata."""
from pathlib import Path
from typing import Any
try:
    import fitz
except ImportError:
    fitz = None

def load_pdf(path: str) -> list[dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDF ingestion")
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(path)
    doc=fitz.open(str(p))
    return [{"text": page.get_text("text"), "document": p.name, "page": i+1,
             "metadata":{"source_path":str(p)}} for i,page in enumerate(doc)]

def load_documents(paths: list[str]) -> list[dict[str, Any]]:
    pages=[]
    for path in paths: pages.extend(load_pdf(path))
    return pages
