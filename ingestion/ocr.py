"""OCR fallback for image-only or near-empty PDF pages."""
from typing import Any
def needs_ocr(text:str,min_chars:int=40)->bool: return len((text or "").strip())<min_chars
def ocr_page(page:Any,language:str="eng",dpi:int=200)->str:
    try:
        import fitz,pytesseract
        from PIL import Image
    except ImportError as exc: raise RuntimeError("OCR requires PyMuPDF, pytesseract, and Pillow") from exc
    pix=page.get_pixmap(matrix=fitz.Matrix(dpi/72.0,dpi/72.0),alpha=False)
    return pytesseract.image_to_string(Image.frombytes("RGB",(pix.width,pix.height),pix.samples),lang=language).strip()