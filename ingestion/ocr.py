"""Optional OCR fallback for scanned PDFs."""
from typing import Any

def ocr_page(page: Any, language: str="eng") -> str:
    try:
        import pytesseract
        from PIL import Image
        pix=page.get_pixmap(matrix=__import__("fitz").Matrix(2,2))
        img=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
        return pytesseract.image_to_string(img, lang=language)
    except ImportError as exc:
        raise RuntimeError("Install pytesseract and Pillow for OCR") from exc

def needs_ocr(text: str, min_chars: int=20) -> bool:
    return len((text or "").strip()) < min_chars
