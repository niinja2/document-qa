"""
/upload endpoint tests — CAT-01 through CAT-04
Covers: file type, file count, file size, session UUID.
"""
import io
import uuid

import pytest
import httpx

from conftest import upload_file


def _upload_expect_4xx(client, *args, **kwargs):
    """Upload and assert 4xx. Treats a connection drop (server crash) as a failure."""
    try:
        r = upload_file(client, *args, **kwargs)
        assert 400 <= r.status_code < 500, (
            f"Expected 4xx but got {r.status_code} — server should return a clean error"
        )
    except httpx.ReadError:
        pytest.fail(
            "Server dropped the connection (WinError 10054) instead of returning 422. "
            "Unhandled exception in the server — add error handling."
        )


# ── CAT-01: File Type ─────────────────────────────────────────────────────────

class TestFileType:

    def test_T001_pdf_with_text_layer(self, client, session_id, text_pdf_bytes):
        """T001 — PDF with selectable text → 200, text extracted."""
        r = upload_file(client, session_id, text_pdf_bytes, "sample.pdf")
        assert r.status_code == 200

    def test_T002_png_image(self, client, session_id, text_png_bytes):
        """T002 — PNG with printed text → 200, OCR applied."""
        r = upload_file(client, session_id, text_png_bytes, "sample.png", "image/png")
        assert r.status_code == 200

    def test_T003_jpg_image(self, client, session_id, text_jpg_bytes):
        """T003 — JPG with printed text → 200, OCR applied."""
        r = upload_file(client, session_id, text_jpg_bytes, "sample.jpg", "image/jpeg")
        assert r.status_code == 200

    def test_T004_exe_rejected(self, client, session_id):
        """T004 — .exe binary → 4xx unsupported type error."""
        r = upload_file(client, session_id, b"MZ\x90\x00fake", "malware.exe", "application/octet-stream")
        assert r.status_code >= 400

    def test_T005_docx_rejected(self, client, session_id):
        """T005 — .docx (not in spec) → 422. Bug if server crashes instead."""
        _upload_expect_4xx(
            client, session_id, b"PK\x03\x04fakecontent", "doc.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_T006_csv_rejected(self, client, session_id):
        """T006 — .csv plaintext → 4xx unsupported type error."""
        r = upload_file(client, session_id, b"id,name\n1,Alice", "data.csv", "text/csv")
        assert r.status_code >= 400

    def test_supported_and_unsupported_produce_different_status(
        self, client, text_pdf_bytes, text_png_bytes
    ):
        """Opposite inputs must yield different outcomes: supported≠unsupported."""
        sid_good = str(uuid.uuid4())
        sid_bad  = str(uuid.uuid4())
        good = upload_file(client, sid_good, text_pdf_bytes, "ok.pdf")
        bad  = upload_file(client, sid_bad,  b"MZ\x90fake", "bad.exe", "application/octet-stream")
        assert good.status_code == 200
        assert bad.status_code >= 400
        assert good.status_code != bad.status_code


# ── CAT-02: File Count ────────────────────────────────────────────────────────

class TestFileCount:

    def test_T007_no_file_field(self, client, session_id):
        """T007 — No file field in multipart → 422 validation error."""
        r = client.post("/upload", data={"session_id": session_id})
        assert r.status_code == 422

    def test_T008_one_file(self, client, session_id, text_pdf_bytes):
        """T008 — Exactly one file → 200."""
        r = upload_file(client, session_id, text_pdf_bytes, "a.pdf")
        assert r.status_code == 200

    def test_T009_sequential_uploads_same_session(self, client, session_id, text_pdf_bytes, text_png_bytes):
        """T009 — Two sequential single-file uploads to same session → both 200."""
        r1 = upload_file(client, session_id, text_pdf_bytes, "a.pdf")
        r2 = upload_file(client, session_id, text_png_bytes, "b.png", "image/png")
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_T010_many_sequential_uploads(self, client, session_id, text_pdf_bytes):
        """T010 — Five sequential uploads to same session → all 200, no crash."""
        for i in range(5):
            r = upload_file(client, session_id, text_pdf_bytes, f"doc{i}.pdf")
            assert r.status_code == 200, f"Upload {i} failed with {r.status_code}"


# ── CAT-03: File Size ─────────────────────────────────────────────────────────

class TestFileSize:

    def test_T011_empty_file(self, client, session_id):
        """T011 — 0-byte file → 422 (contract: empty text = parse error). Bug if server 500s."""
        _upload_expect_4xx(client, session_id, b"", "empty.pdf")

    def test_T012_corrupt_one_byte(self, client, session_id):
        """T012 — 1-byte corrupt file → 422 (contract: parse error). Bug if server crashes."""
        _upload_expect_4xx(client, session_id, b"\x00", "corrupt.pdf")

    def test_T013_normal_size(self, client, session_id, text_pdf_bytes):
        """T013 — Normal PDF → 200."""
        r = upload_file(client, session_id, text_pdf_bytes, "normal.pdf")
        assert r.status_code == 200


# ── CAT-04: Session UUID ──────────────────────────────────────────────────────

class TestSessionUUID:

    def test_T016_missing_session_id_field(self, client, text_pdf_bytes):
        """T016 — No session_id in multipart → 422 validation error. Bug if server crashes."""
        try:
            r = client.post(
                "/upload",
                files={"files": ("a.pdf", io.BytesIO(text_pdf_bytes), "application/pdf")},
            )
            assert r.status_code == 422, (
                f"Expected 422 for missing session_id, got {r.status_code}"
            )
        except httpx.ReadError:
            pytest.fail(
                "Server dropped the connection (WinError 10054) instead of returning 422. "
                "Missing session_id field causes unhandled exception — add validation."
            )

    def test_T017_empty_string_session_id(self, client, text_pdf_bytes):
        """T017 — Empty string session_id → rejected (4xx)."""
        r = upload_file(client, "", text_pdf_bytes, "a.pdf")
        assert r.status_code >= 400

    def test_T018_valid_uuid4(self, client, text_pdf_bytes):
        """T018 — Well-formed UUID v4 → 200."""
        sid = str(uuid.uuid4())
        r = upload_file(client, sid, text_pdf_bytes, "a.pdf")
        assert r.status_code == 200

    def test_T019_malformed_session_id(self, client, text_pdf_bytes):
        """T019 — Malformed session_id string → document expected behaviour.

        The spec doesn't state the exact behaviour; we assert it does NOT 500.
        """
        r = upload_file(client, "not-a-uuid", text_pdf_bytes, "a.pdf")
        assert r.status_code != 500

    def test_T020_re_upload_same_session(self, client, session_id, text_pdf_bytes):
        """T020 — Upload to same UUID twice → 200 both times (no 409 conflict)."""
        r1 = upload_file(client, session_id, text_pdf_bytes, "first.pdf")
        r2 = upload_file(client, session_id, text_pdf_bytes, "second.pdf")
        assert r1.status_code == 200
        assert r2.status_code == 200


# ── CAT-05: Mutation tests ────────────────────────────────────────────────────

class TestMutations:
    """Each test mutates exactly one property of a valid request and checks the result."""

    # ── File content/extension mismatch ──

    def test_MUT01_pdf_extension_image_bytes(self, client, session_id, text_png_bytes):
        """MUT01 — PNG bytes sent with .pdf filename → server validates by content, not extension.

        If the server checks extension only: accepts (wrong). If it checks magic bytes: rejects.
        Either way it must not 500.
        """
        r = upload_file(client, session_id, text_png_bytes, "fake.pdf", "application/pdf")
        assert r.status_code != 500

    def test_MUT02_image_extension_pdf_bytes(self, client, session_id, text_pdf_bytes):
        """MUT02 — PDF bytes sent with .png filename → server validates by content, not extension.

        Must not crash regardless of which validation strategy the server uses.
        """
        r = upload_file(client, session_id, text_pdf_bytes, "fake.png", "image/png")
        assert r.status_code != 500

    def test_MUT03_pdf_bytes_wrong_content_type(self, client, session_id, text_pdf_bytes):
        """MUT03 — Valid PDF bytes but Content-Type claims image/png → no crash.

        Tests whether server trusts Content-Type header or inspects actual bytes.
        """
        r = upload_file(client, session_id, text_pdf_bytes, "doc.pdf", "image/png")
        assert r.status_code != 500

    def test_MUT04_valid_header_truncated_body(self, client, session_id):
        """MUT04 — File starts with valid %PDF header but body is truncated garbage.

        Passes the size guard (>4 bytes) but should fail extraction gracefully → 4xx not 500.
        """
        truncated = b"%PDF-1.4\n" + b"\x00\xff" * 10
        _upload_expect_4xx(client, session_id, truncated, "truncated.pdf")

    # ── Session ID mutations ──

    def test_MUT05_uuid_uppercase(self, client, session_id, text_pdf_bytes):
        """MUT05 — UUID with uppercase hex letters → accepted (UUIDs are case-insensitive)."""
        upper_uuid = str(uuid.uuid4()).upper()
        r = upload_file(client, upper_uuid, text_pdf_bytes, "a.pdf")
        assert r.status_code != 500

    def test_MUT06_uuid_with_extra_suffix(self, client, text_pdf_bytes):
        """MUT06 — UUID with extra characters appended → 4xx or accepted, not 500."""
        bad_uuid = str(uuid.uuid4()) + "EXTRA"
        r = upload_file(client, bad_uuid, text_pdf_bytes, "a.pdf")
        assert r.status_code != 500

    def test_MUT07_duplicate_session_id_fields(self, client, text_pdf_bytes):
        """MUT07 — Two session_id values in the same multipart → no crash.

        FastAPI will typically take the first or last value; must not raise 500.
        """
        sid = str(uuid.uuid4())
        r = client.post(
            "/upload",
            files=[
                ("session_id", (None, sid, None)),
                ("session_id", (None, str(uuid.uuid4()), None)),
                ("file", ("a.pdf", io.BytesIO(text_pdf_bytes), "application/pdf")),
            ],
        )
        assert r.status_code != 500

    # ── Question mutations ──

    def test_MUT08_null_byte_in_question(self, client, session_id, text_pdf_bytes):
        """MUT08 — Null byte embedded in question string → no crash."""
        from conftest import ask
        upload_file(client, session_id, text_pdf_bytes, "doc.pdf")
        r = ask(client, session_id, "question" + "\x00" + "injection")
        assert r.status_code != 500

    def test_MUT09_unicode_control_chars_in_question(self, client, session_id, text_pdf_bytes):
        """MUT09 — Unicode control characters in question → no crash."""
        from conftest import ask
        upload_file(client, session_id, text_pdf_bytes, "doc.pdf")
        r = ask(client, session_id, "What" + chr(0) + "is" + chr(0x1f) + "this\u200b document?")
        assert r.status_code != 500

    def test_MUT10_rtl_unicode_question(self, client, session_id, text_pdf_bytes):
        """MUT10 — Right-to-left Unicode question (Arabic) → no crash."""
        from conftest import ask
        upload_file(client, session_id, text_pdf_bytes, "doc.pdf")
        r = ask(client, session_id, "ما هو موضوع هذه الوثيقة؟")
        assert r.status_code != 500

    def test_MUT11_numbers_only_question(self, client, session_id, text_pdf_bytes):
        """MUT11 — Question is only digits and symbols, no words → no crash."""
        from conftest import ask
        upload_file(client, session_id, text_pdf_bytes, "doc.pdf")
        r = ask(client, session_id, "42 + 3.14 = ???")
        assert r.status_code != 500

    def test_MUT12_extra_field_in_upload(self, client, session_id, text_pdf_bytes):
        """MUT12 — Extra unexpected field in multipart → ignored or rejected, not 500."""
        r = client.post(
            "/upload",
            data={"session_id": session_id, "unexpected_field": "surprise"},
            files={"file": ("a.pdf", io.BytesIO(text_pdf_bytes), "application/pdf")},
        )
        assert r.status_code != 500
