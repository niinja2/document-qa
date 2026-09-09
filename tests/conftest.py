"""
Shared fixtures for Document QA test suite.

Assumptions (spec-only, no code read):
  - FastAPI server at http://localhost:8000
  - POST /upload  multipart/form-data  fields: file, session_id (str)
  - POST /ask     form-data            fields: session_id (str), question (str)
  - Success responses: JSON with at least {"answer": str} on /ask
"""
import io
import os
import sys
import uuid

import pytest
import httpx

# Load .env so unit tests that import config.py see the same values as the server.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # python-dotenv not installed — config.py falls back to its own defaults

BASE_URL = "http://localhost:8000"


# ── HTTP client ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as c:
        yield c


# ── Session identifiers ───────────────────────────────────────────────────────

@pytest.fixture
def session_id():
    return str(uuid.uuid4())


@pytest.fixture
def fresh_session_id():
    """UUID that has never been used to upload anything."""
    return str(uuid.uuid4())


# ── PDF fixtures (built with fitz / PyMuPDF which is a project dependency) ──

@pytest.fixture(scope="session")
def text_pdf_bytes():
    """Minimal PDF with a readable text layer."""
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page()
    sentence = "The quick brown fox jumps over the lazy dog. "
    page.insert_text((72, 72), sentence * 30)  # ~270 words
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="session")
def large_text_pdf_bytes():
    """PDF with >500 words to exercise chunker boundary."""
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page()
    # ~600 distinct words
    words = " ".join(f"word{i}" for i in range(600))
    page.insert_text((72, 72), words, fontsize=6)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="session")
def blank_pdf_bytes():
    """PDF with pages but no content (empty text layer, no images)."""
    import pymupdf as fitz
    doc = fitz.open()
    doc.new_page()  # blank page, nothing inserted
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="session")
def corrupt_pdf_bytes():
    """One byte — not a valid PDF."""
    return b"\x00"


@pytest.fixture(scope="session")
def text_png_bytes():
    """PNG image with clear black text for EasyOCR."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 80), "Sample OCR document content for testing", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def text_jpg_bytes():
    """JPEG image with clear black text for EasyOCR."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 80), "JPEG OCR test content document", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── Real test docs (committed to repo) ───────────────────────────────────────

@pytest.fixture(scope="session")
def benchmark_pdf_path():
    return "test_docs/Embedding Model Retrieval Benchmark.pdf"


@pytest.fixture(scope="session")
def design_notes_pdf_path():
    return "test_docs/Retrieval_Quality_Design_Notes.pdf"


# ── Helpers ───────────────────────────────────────────────────────────────────

def upload_file(client, session_id, file_bytes, filename, content_type="application/pdf"):
    """POST /upload with a single in-memory file."""
    return client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": (filename, io.BytesIO(file_bytes), content_type)},
    )


def ask(client, session_id, question):
    """POST /ask."""
    return client.post("/ask", data={"session_id": session_id, "question": question})
