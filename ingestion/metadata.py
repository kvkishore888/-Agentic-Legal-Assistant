"""Stable source metadata helpers."""
import re
from pathlib import Path

def safe_document_name(name: str) -> str:
    return Path(name).name

def make_chunk_id(document: str, page: int, index: int) -> str:
    stem=re.sub(r"[^A-Za-z0-9]+","_",Path(document).stem).strip("_") or "document"
    return f"{stem}_p{page}_c{index}"

def build_metadata(document: str, page: int, section: str="", **extra) -> dict:
    return {"document":safe_document_name(document),"page":int(page),
            "section":section or "", **extra}
