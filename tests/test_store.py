"""
Unit tests for session/store.py — CAT-11
RAM dict keyed by session UUID; no persistence across restarts.
"""
import sys
import os
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from session.store import save, load

# Contract: save(session_id: str, chunks: list[str], index) -> None
#           load(session_id: str) -> {"chunks": [...], "index": ...} | None

DUMMY_CHUNKS = [{"text": "chunk one", "doc_id": "a.pdf"}, {"text": "chunk two", "doc_id": "a.pdf"}]
DUMMY_INDEX  = object()  # placeholder for a FAISS index


class TestSessionStore:

    def test_T046_save_then_load_returns_same_data(self):
        """T046 — save then load → dict with same chunks and index."""
        sid = str(uuid.uuid4())
        save(sid, DUMMY_CHUNKS, DUMMY_INDEX)
        result = load(sid)
        assert result is not None
        assert result["chunks"] is DUMMY_CHUNKS
        assert result["index"] is DUMMY_INDEX

    def test_T047_load_unknown_uuid(self):
        """T047 — load(never-saved UUID) → None (not a KeyError crash)."""
        sid = str(uuid.uuid4())
        try:
            result = load(sid)
            assert result is None, f"Expected None for unknown UUID, got {result!r}"
        except KeyError:
            pytest.fail("store.load raised raw KeyError — should return None")

    def test_T048_overwrite_same_uuid(self):
        """T048 — save twice with same UUID → second save replaces first."""
        sid = str(uuid.uuid4())
        chunks_v1 = [{"text": "old", "doc_id": "a.pdf"}]
        chunks_v2 = [{"text": "new", "doc_id": "b.pdf"}]
        index_v1 = object()
        index_v2 = object()
        save(sid, chunks_v1, index_v1)
        save(sid, chunks_v2, index_v2)
        result = load(sid)
        assert result["chunks"] is chunks_v2
        assert result["index"] is index_v2

    def test_different_uuids_do_not_collide(self):
        """Distinct UUIDs must map to distinct data slots."""
        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())
        chunks_a = [{"text": "alpha", "doc_id": "a.pdf"}]
        chunks_b = [{"text": "beta",  "doc_id": "b.pdf"}]
        save(sid_a, chunks_a, None)
        save(sid_b, chunks_b, None)
        assert load(sid_a)["chunks"] is chunks_a
        assert load(sid_b)["chunks"] is chunks_b

    def test_uuid_string_accepted(self):
        """store must accept the UUID as a plain string."""
        sid = str(uuid.uuid4())
        save(sid, DUMMY_CHUNKS, DUMMY_INDEX)
        assert load(sid) is not None
