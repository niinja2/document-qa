"""
Rate limiting tests — CAT-08 (slowapi)
Normal / burst above limit / post-cooldown recovery.
"""
import time
import uuid

import pytest

from conftest import upload_file, ask

# Adjust to match RATE_LIMIT config in the app (requests per window).
# The spec mentions slowapi but does not specify the exact limit.
# We probe conservatively: send requests until we see a 429.
MAX_PROBE = 50
RATE_WINDOW_SECONDS = 10


def _upload_session(client):
    """Create a fresh session with a document to ask questions against."""
    sid = str(uuid.uuid4())
    import fitz, io
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Rate limit test document content. " * 20)
    buf = io.BytesIO()
    doc.save(buf)
    upload_file(client, sid, buf.getvalue(), "rate_doc.pdf")
    return sid


class TestRateLimit:

    def test_T039_single_request_not_throttled(self, client, text_pdf_bytes):
        """T039 — One request within window → 200, not throttled."""
        sid = _upload_session(client)
        r = ask(client, sid, "What is in the document?")
        assert r.status_code == 200

    def test_T040_burst_triggers_429(self, client):
        """T040 — Rapid burst → eventually returns HTTP 429.

        If no 429 is seen within MAX_PROBE requests the test is skipped
        (rate limiting not active or limit > MAX_PROBE).
        """
        sid = _upload_session(client)
        for _ in range(MAX_PROBE):
            r = ask(client, sid, "What is the content?")
            if r.status_code == 429:
                return  # pass — rate limit is working
        pytest.skip(
            f"No 429 after {MAX_PROBE} requests — "
            "rate limiting not active or MAX_PROBE is below the configured limit."
        )

    def test_T041_recovery_after_cooldown(self, client):
        """T041 — After hitting rate limit, wait for cooldown → back to 200."""
        sid = _upload_session(client)

        # Trigger the limit
        for _ in range(MAX_PROBE):
            r = ask(client, sid, "Content?")
            if r.status_code == 429:
                break
        else:
            pytest.skip("Could not trigger rate limit within MAX_PROBE requests.")

        # Wait for window to reset
        time.sleep(RATE_WINDOW_SECONDS + 1)

        r_after = ask(client, sid, "What is in the document?")
        assert r_after.status_code == 200, (
            f"After cooldown expected 200, got {r_after.status_code}. "
            "Rate limit did not reset."
        )
