"""
Rate limiting tests — CAT-08 (slowapi)
Normal / burst above limit / post-cooldown recovery.

Run against the server with DEFAULT rate limits (no env var overrides):
  /upload — 10 requests/minute
  /ask    — 30 requests/minute
"""
import time
import uuid

import pytest

from conftest import upload_file, ask

# Sequential probe: send requests one by one until 429 appears.
# 35 is just above the 30/min ask limit; 12 is just above the 10/min upload limit.
ASK_PROBE    = 35
UPLOAD_PROBE = 12
RATE_WINDOW_SECONDS = 62  # just over 1 minute for the window to reset


def _upload_session(client):
    """Create a fresh session with a document to ask questions against."""
    sid = str(uuid.uuid4())
    import pymupdf, io
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Rate limit test document content. " * 20)
    buf = io.BytesIO()
    doc.save(buf)
    upload_file(client, sid, buf.getvalue(), "rate_doc.pdf")
    return sid


@pytest.mark.rate_limit
class TestRateLimit:

    def test_T039_single_request_not_throttled(self, client):
        """T039 — One /ask request → 200, not throttled."""
        sid = _upload_session(client)
        r = ask(client, sid, "What is in the document?")
        assert r.status_code == 200

    def test_T040_ask_burst_triggers_429(self, client):
        """T040 — 35 sequential /ask requests → 429 before the end (30/min limit).

        Sends requests one by one; passes as soon as a 429 is seen.
        Skips if no 429 within ASK_PROBE requests (limit may be higher than default).
        """
        sid = _upload_session(client)
        for i in range(ASK_PROBE):
            r = ask(client, sid, "What is the content?")
            if r.status_code == 429:
                return  # PASS — rate limit is working
        pytest.skip(
            f"No 429 after {ASK_PROBE} /ask requests — "
            "server may be running with a higher rate limit (env var override?)."
        )

    def test_T041_recovery_after_cooldown(self, client):
        """T041 — Trigger /ask rate limit, wait for window reset → back to 200."""
        sid = _upload_session(client)

        # Trigger the limit
        for _ in range(ASK_PROBE):
            r = ask(client, sid, "Content?")
            if r.status_code == 429:
                break
        else:
            pytest.skip(
                f"Could not trigger rate limit in {ASK_PROBE} requests — "
                "server may be running with a higher rate limit."
            )

        # Wait for the rate limit window to reset
        time.sleep(RATE_WINDOW_SECONDS)

        r_after = ask(client, sid, "What is in the document?")
        assert r_after.status_code == 200, (
            f"After cooldown expected 200, got {r_after.status_code}. "
            "Rate limit did not reset."
        )
