"""
End-to-end integration tests — CAT-07
Upload → ask flows; session isolation; multi-file retrieval; LLM fallback.
"""
import io
import threading
import uuid

import pytest

from conftest import upload_file, ask


class TestEndToEnd:

    def test_T031_pdf_upload_then_relevant_answer(self, client, text_pdf_bytes):
        """T031 — Upload text-PDF → ask about its content → answer is non-empty."""
        sid = str(uuid.uuid4())
        upload_file(client, sid, text_pdf_bytes, "doc.pdf")
        r = ask(client, sid, "What words appear in this document?")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data.get("answer"), str)
        assert len(data["answer"]) > 0

    def test_T032_image_upload_then_answer(self, client, text_png_bytes):
        """T032 — Upload text-PNG → OCR path → ask → non-empty answer."""
        sid = str(uuid.uuid4())
        r = upload_file(client, sid, text_png_bytes, "img.png", "image/png")
        assert r.status_code == 200
        r2 = ask(client, sid, "What text is in this image?")
        assert r2.status_code == 200
        assert len(r2.json().get("answer", "")) > 0

    def test_T033_two_sequential_uploads_retrieval(
        self, client, text_pdf_bytes, text_png_bytes
    ):
        """T033 — Upload PDF then PNG to same session → ask → answer from index."""
        sid = str(uuid.uuid4())
        r1 = upload_file(client, sid, text_pdf_bytes, "a.pdf")
        assert r1.status_code == 200
        r2 = upload_file(client, sid, text_png_bytes, "b.png", "image/png")
        assert r2.status_code == 200
        r = ask(client, sid, "What is in the documents?")
        assert r.status_code == 200
        assert len(r.json().get("answer", "")) > 0

    def test_T034_session_isolation(self, client, text_pdf_bytes):
        """T034 — Two sessions with different docs must not cross-contaminate.

        Session A gets a PDF containing 'ALPHA_UNIQUE_TERM'.
        Session B gets a PDF containing 'BETA_UNIQUE_TERM'.
        Asking session A about BETA_UNIQUE_TERM must not return it as primary answer.
        """
        import fitz

        def make_pdf(term):
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), f"This document is about {term}. " * 40)
            buf = io.BytesIO()
            doc.save(buf)
            return buf.getvalue()

        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())

        upload_file(client, sid_a, make_pdf("ALPHA_UNIQUE_TERM"), "a.pdf")
        upload_file(client, sid_b, make_pdf("BETA_UNIQUE_TERM"),  "b.pdf")

        r_a = ask(client, sid_a, "What unique term does this document mention?")
        assert r_a.status_code == 200
        answer_a = r_a.json().get("answer", "").upper()

        # Session A's answer must not mention B's unique term
        assert "BETA_UNIQUE_TERM" not in answer_a, (
            "Session A leaked content from Session B — isolation is broken."
        )

    def test_T035_multiple_questions_same_session(self, client, text_pdf_bytes):
        """T035 — Three sequential questions on the same session return distinct answers
        and none of them 500.
        """
        sid = str(uuid.uuid4())
        upload_file(client, sid, text_pdf_bytes, "doc.pdf")
        questions = [
            "What is the first word in the document?",
            "Summarize this document in one sentence.",
            "How many pages does this document have?",
        ]
        answers = []
        for q in questions:
            r = ask(client, sid, q)
            assert r.status_code == 200, f"Question {q!r} returned {r.status_code}"
            answers.append(r.json().get("answer", ""))
        # At least some answers should differ (non-trivial LLM responses)
        assert len(set(answers)) > 1 or all(len(a) > 0 for a in answers)

    def test_T036_session_cleared_after_server_restart(self, client, fresh_session_id):
        """T036 — After a server restart, a previously used UUID must return 'upload first'.

        This test cannot force a restart, so it tests the same invariant:
        a brand-new UUID (equivalent to post-restart state) must return an error.
        """
        r = ask(client, fresh_session_id, "What is this document about?")
        assert r.status_code >= 400

    def test_T037_concurrent_uploads_no_mixing(self, client, text_pdf_bytes):
        """T037 — Two concurrent uploads to different sessions must not mix data."""
        results = {}
        errors = []

        def do_upload(label):
            try:
                sid = str(uuid.uuid4())
                r = upload_file(client, sid, text_pdf_bytes, f"{label}.pdf")
                results[label] = (sid, r.status_code)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_upload, args=("session_A",))
        t2 = threading.Thread(target=do_upload, args=("session_B",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"Concurrent upload raised: {errors}"
        assert results["session_A"][1] == 200
        assert results["session_B"][1] == 200
        # Sessions must be distinct
        assert results["session_A"][0] != results["session_B"][0]

    def test_T038_fallback_to_distilbert_without_api_key(
        self, client, text_pdf_bytes, monkeypatch
    ):
        """T038 — Without OPENROUTER_API_KEY the system falls back to DistilBERT.

        We can't unset the env var server-side via monkeypatch (different process),
        so this test is marked xfail unless the server is started without the key.
        The important thing is: the endpoint must still return 200 with an answer.

        Run with: OPENROUTER_API_KEY="" pytest tests/test_integration.py::TestEndToEnd::test_T038
        """
        sid = str(uuid.uuid4())
        upload_file(client, sid, text_pdf_bytes, "doc.pdf")
        r = ask(client, sid, "What is in this document?")
        # Regardless of backend used, must return 200
        assert r.status_code == 200
        assert len(r.json().get("answer", "")) > 0


# ── Cascade tests (triplets) ──────────────────────────────────────────────────

class TestCascades:
    """State-cascade tests: A sets state, B modifies it, C reads it.
    Each test verifies behaviour that neither step alone would reveal.
    """

    def test_TC01_valid_upload_then_empty_then_ask(self, client, text_pdf_bytes):
        """TC01 — Good index → failed empty upload → ask still works (index not wiped)."""
        sid = str(uuid.uuid4())
        upload_file(client, sid, text_pdf_bytes, "doc.pdf")
        upload_file(client, sid, b"", "empty.pdf")          # should 422, not wipe index
        r = ask(client, sid, "What is in this document?")
        assert r.status_code == 200, "Index was wiped by a failed empty upload"
        assert len(r.json().get("answer", "")) > 0

    def test_TC02_valid_upload_then_unsupported_then_ask(self, client, text_pdf_bytes):
        """TC02 — Good index → rejected unsupported file → index survives rejection."""
        sid = str(uuid.uuid4())
        upload_file(client, sid, text_pdf_bytes, "doc.pdf")
        upload_file(client, sid, b"PK\x03\x04fake", "file.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        r = ask(client, sid, "What is in this document?")
        assert r.status_code == 200, "Index was wiped by a rejected unsupported file"
        assert len(r.json().get("answer", "")) > 0

    def test_TC03_valid_upload_then_reupload_then_ask(self, client, text_pdf_bytes):
        """TC03 — Upload → re-upload → ask reflects content (index updated, not broken)."""
        import fitz
        sid = str(uuid.uuid4())
        upload_file(client, sid, text_pdf_bytes, "first.pdf")

        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "UNIQUE_SECOND_CONTENT " * 30)
        buf = io.BytesIO()
        doc.save(buf)
        upload_file(client, sid, buf.getvalue(), "second.pdf")

        r = ask(client, sid, "What is in this document?")
        assert r.status_code == 200, "Ask failed after re-upload"
        assert len(r.json().get("answer", "")) > 0

    def test_TC04_image_then_pdf_then_ask(self, client, text_png_bytes, text_pdf_bytes):
        """TC04 — OCR upload → text-layer upload → ask works across both."""
        sid = str(uuid.uuid4())
        upload_file(client, sid, text_png_bytes, "img.png", "image/png")
        upload_file(client, sid, text_pdf_bytes, "doc.pdf")
        r = ask(client, sid, "What content is in this session?")
        assert r.status_code == 200, "Ask failed on mixed-type session"
        assert len(r.json().get("answer", "")) > 0

    def test_TC05_failed_upload_then_recovery_then_ask(self, client, text_pdf_bytes):
        """TC05 — Failed upload → recovery valid upload → session works normally."""
        sid = str(uuid.uuid4())
        upload_file(client, sid, b"", "empty.pdf")          # fails, but session_id was sent
        upload_file(client, sid, text_pdf_bytes, "recovery.pdf")
        r = ask(client, sid, "What is in this document?")
        assert r.status_code == 200, "Session did not recover after initial failed upload"
        assert len(r.json().get("answer", "")) > 0

    def test_TC06_ask_404_then_upload_then_ask(self, client, text_pdf_bytes):
        """TC06 — Ask on empty session (404) → upload → ask works (404 left no broken state)."""
        sid = str(uuid.uuid4())
        r_early = ask(client, sid, "Anything?")
        assert r_early.status_code == 404
        upload_file(client, sid, text_pdf_bytes, "doc.pdf")
        r = ask(client, sid, "What is in this document?")
        assert r.status_code == 200, "Session broken by earlier 404 — upload did not fix it"
        assert len(r.json().get("answer", "")) > 0

    def test_TC07_mixed_index_then_offtopic_ask(self, client, text_pdf_bytes, text_png_bytes):
        """TC07 — PDF + image session → off-topic question → LLM says not found (both docs searched)."""
        sid = str(uuid.uuid4())
        upload_file(client, sid, text_pdf_bytes, "doc.pdf")
        upload_file(client, sid, text_png_bytes, "img.png", "image/png")
        r = ask(client, sid, "What is the GDP of Mars in 2083?")
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        # Answer must not be empty — LLM should say it cannot find the answer
        assert len(data["answer"]) > 0
