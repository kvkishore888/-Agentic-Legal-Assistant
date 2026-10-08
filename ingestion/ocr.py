"""OCR fallback for image-only or sparse PDF pages."""
from __future__ import annotations
from typing import Any

def needs_ocr(text: str, min_chars: int = 40) -> bool:
    return len((text or "").strip()) < min_chars

def ocr_page(page: Any, language: str = "eng", dpi: int = 200) -> str:
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("OCR requires PyMuPDF, pytesseract, and Pillow") from exc
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    try:
        return pytesseract.image_to_string(image, lang=language).strip()
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as exc:
        raise RuntimeError("OCR engine unavailable or failed") from exc
