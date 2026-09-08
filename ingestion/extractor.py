import logging
import threading
import pymupdf
import easyocr
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

_ocr_reader = None
_ocr_lock = threading.Lock()


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                _ocr_reader = easyocr.Reader(["en"])
    return _ocr_reader


def _ocr_from_bytes(image_bytes: bytes) -> str:
    try:
        reader = _get_ocr_reader()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        results = reader.readtext(image_np, detail=0)
        return "\n".join(results)
    except Exception:
        return ""


def extract_pdf(file_path: str) -> str:
    try:
        with pymupdf.open(file_path) as doc:
            pages = []
            for page in doc:
                page_text = page.get_text()
                image_texts = []
                for img in page.get_images():
                    xref = img[0]
                    image_data = doc.extract_image(xref)
                    ocr_text = _ocr_from_bytes(image_data["image"])
                    if ocr_text:
                        image_texts.append(ocr_text)
                if image_texts:
                    combined = page_text + "\n" + "\n".join(image_texts)
                else:
                    combined = page_text
                pages.append(combined)
        return "\n".join(pages)
    except Exception as e:
        raise ValueError(f"Could not read PDF file: {e}")


def extract_image(file_path: str) -> str:
    try:
        reader = _get_ocr_reader()
        results = reader.readtext(file_path, detail=0)
        return "\n".join(results)
    except Exception as e:
        raise ValueError(f"Could not read image file: {e}")


def _is_pdf(file_path: str) -> bool:
    with open(file_path, "rb") as f:
        return f.read(4) == b"%PDF"


def extract(file_path: str) -> str:
    path = file_path.lower()
    if _is_pdf(file_path):
        text = extract_pdf(file_path)
    elif path.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        text = extract_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    logger.info("Extracted %d characters from %s", len(text), file_path)
    return text
