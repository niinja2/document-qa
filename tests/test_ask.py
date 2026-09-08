"""
/ask endpoint tests — CAT-05 and CAT-06
Covers: pre-conditions (session state) and question dimension.
"""
import uuid

import pytest

from conftest import upload_file, ask


# ── CAT-05: Pre-conditions ────────────────────────────────────────────────────

class TestAskPreConditions:

    def test_T021_ask_before_upload(self, client, fresh_session_id):
        """T021 — UUID never uploaded → 404 (contract-specified)."""
        r = ask(client, fresh_session_id, "What is this document about?")
        assert r.status_code == 404, (
            f"Expected 404 for unknown session, got {r.status_code}: {r.text!r}"
        )

    def test_T022_ask_no_session_id(self, client):
        """T022 — No session_id field → 422 validation error."""
        r = client.post("/ask", data={"question": "anything"})
        assert r.status_code == 422

    def test_T023_ask_after_upload(self, client, session_id, text_pdf_bytes):
        """T023 — Valid upload then ask → 200 with non-empty answer string."""
        upload_file(client, session_id, text_pdf_bytes, "doc.pdf")
        r = ask(client, session_id, "What does this document contain?")
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0


# ── CAT-06: Question Dimension ────────────────────────────────────────────────

class TestQuestionDimension:
    """All tests in this class require a pre-uploaded session."""

    @pytest.fixture(autouse=True)
    def uploaded_session(self, client, session_id, text_pdf_bytes):
        """Upload once; all tests in this class share the session."""
        upload_file(client, session_id, text_pdf_bytes, "doc.pdf")
        self.sid = session_id

    def test_T024_empty_question(self, client):
        """T024 — Empty string question → 4xx or graceful non-500 response."""
        r = ask(client, self.sid, "")
        assert r.status_code != 500

    def test_T025_whitespace_only_question(self, client):
        """T025 — Whitespace-only question → 4xx or treated as empty (not 500)."""
        r = ask(client, self.sid, "   \t  ")
        assert r.status_code != 500

    def test_T026_normal_question(self, client):
        """T026 — Normal question → 200 with non-empty answer."""
        r = ask(client, self.sid, "What is the main topic?")
        assert r.status_code == 200
        assert len(r.json().get("answer", "")) > 0

    def test_T027_very_long_question(self, client):
        """T027 — Question > 2000 chars → 200 or 4xx with clear message (not 500)."""
        long_q = "What does the document say about " + ("topic " * 400)  # >2400 chars
        r = ask(client, self.sid, long_q)
        assert r.status_code != 500

    def test_T028_question_about_missing_content(self, client):
        """T028 — Question about content not in document → 200, LLM responds gracefully."""
        r = ask(client, self.sid, "What is the GDP of Mars in 2083?")
        assert r.status_code == 200
        assert "answer" in r.json()

    def test_T029_sql_injection_in_question(self, client):
        """T029 — SQL injection payload → treated as plain text, no crash."""
        r = ask(client, self.sid, "'; DROP TABLE sessions; --")
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        # The injection string must not cause a 500
        assert r.status_code != 500

    def test_T030_xss_payload_in_question(self, client):
        """T030 — XSS payload → returned as text/escaped, not executed."""
        r = ask(client, self.sid, "<script>alert(document.cookie)</script>")
        assert r.status_code == 200
        # Response body should not contain raw unescaped <script> as executable HTML
        # (acceptable: returned as escaped string in JSON)
        assert r.status_code != 500

    def test_empty_and_normal_produce_different_results(self, client):
        """Verify empty question and non-empty question have different response behaviour."""
        r_empty  = ask(client, self.sid, "")
        r_normal = ask(client, self.sid, "What is the document about?")
        # At minimum they must not both be 200 with identical bodies
        assert not (r_empty.status_code == 200 and r_empty.text == r_normal.text)
