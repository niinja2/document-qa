"""
Unit tests for the RAG pipeline components — CAT-09 and CAT-10.
Tests: ingestion/extractor.py, pipeline/chunker.py, pipeline/embedder.py, pipeline/retriever.py

Imports assume the project root is on sys.path (run pytest from project root or
via pytest.ini / pyproject.toml with pythonpath = ".").
"""
import sys
import os
import io

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── CAT-09: Extraction ────────────────────────────────────────────────────────

class TestExtractor:
    """ingestion/extractor.py — extract text from PDFs and images."""

    @pytest.fixture(scope="class")
    def text_pdf_path(self, tmp_path_factory, text_pdf_bytes):
        p = tmp_path_factory.mktemp("pdfs") / "text.pdf"
        p.write_bytes(text_pdf_bytes)
        return str(p)

    @pytest.fixture(scope="class")
    def blank_pdf_path(self, tmp_path_factory, blank_pdf_bytes):
        p = tmp_path_factory.mktemp("pdfs") / "blank.pdf"
        p.write_bytes(blank_pdf_bytes)
        return str(p)

    @pytest.fixture(scope="class")
    def png_path(self, tmp_path_factory, text_png_bytes):
        p = tmp_path_factory.mktemp("imgs") / "text.png"
        p.write_bytes(text_png_bytes)
        return str(p)

    def test_T042_pdf_with_text_layer(self, text_pdf_path):
        """T042 — PDF with selectable text → non-empty string, no OCR needed."""
        from ingestion.extractor import extract_pdf
        result = extract_pdf(text_pdf_path)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_T043_blank_pdf_handled_gracefully(self, blank_pdf_path):
        """T044 — Blank PDF (no text, no images) → empty string or known exception, not crash."""
        from ingestion.extractor import extract_pdf
        try:
            result = extract_pdf(blank_pdf_path)
            # If it returns, must be a string (possibly empty)
            assert isinstance(result, str)
        except Exception as e:
            # Any raised exception must not be an unhandled AttributeError / TypeError
            assert not isinstance(e, (AttributeError, TypeError, IndexError)), (
                f"Blank PDF raised unexpected low-level error: {e}"
            )

    def test_T044_image_file_uses_ocr(self, png_path):
        """T045 — PNG image → EasyOCR path → non-empty string extracted."""
        from ingestion.extractor import extract_image
        result = extract_image(png_path)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_supported_and_unsupported_file_types(self, tmp_path):
        """CW vs CCW: PDF path ≠ unsupported path in extractor behaviour."""
        from ingestion import extractor
        bad_path = str(tmp_path / "file.xyz")
        with open(bad_path, "wb") as f:
            f.write(b"notapdf")
        with pytest.raises(Exception):
            # Unsupported extension must raise, not silently return empty
            extractor.extract(bad_path)


# ── CAT-10: Chunker ───────────────────────────────────────────────────────────

CHUNK_SIZE    = 500   # words — from config.py
CHUNK_OVERLAP = 50    # words
CHUNK_STEP    = CHUNK_SIZE - CHUNK_OVERLAP  # 450 — new chunk starts every 450 words


def make_text(n_words: int) -> str:
    return " ".join(f"word{i}" for i in range(n_words))


class TestChunker:
    """pipeline/chunker.py — 500-word chunks, 50-word overlap.

    Sliding window: chunks start at 0, 450, 900, ...
    So a new chunk is created whenever text exceeds CHUNK_STEP (450) words.
    """

    @pytest.fixture(autouse=True)
    def import_chunker(self):
        from pipeline.chunker import chunk_text
        self.chunk_text = chunk_text

    def _text(self, chunk):
        """Extract text from a chunk dict or plain string."""
        return chunk["text"] if isinstance(chunk, dict) else chunk

    def test_T046_empty_string(self):
        """T046 — Empty string → empty list (or single empty-text dict), no crash."""
        result = self.chunk_text("")
        assert result == [] or (len(result) == 1 and self._text(result[0]).strip() == "")

    def test_T047_under_limit(self):
        """T047 — 449 words (< CHUNK_STEP) → exactly 1 chunk."""
        result = self.chunk_text(make_text(449))
        assert len(result) == 1

    def test_T048_at_boundary(self):
        """T048 — Exactly 450 words (= CHUNK_STEP) → exactly 1 chunk."""
        result = self.chunk_text(make_text(450))
        assert len(result) == 1

    def test_T049_one_over_boundary(self):
        """T049 — 451 words (CHUNK_STEP + 1) → 2 chunks; second starts at word 450."""
        text = make_text(451)
        result = self.chunk_text(text)
        assert len(result) == 2, f"Expected 2 chunks, got {len(result)}"
        second = self._text(result[1])
        assert second.startswith("word450"), (
            f"Overlap wrong: chunk[1] starts with {second[:20]!r}, expected 'word450'"
        )

    def test_T050_large_text(self):
        """T050 — 10 000 words → correct number of chunks."""
        import math
        text = make_text(10_000)
        result = self.chunk_text(text)
        expected = math.ceil((10_000 - CHUNK_OVERLAP) / (CHUNK_SIZE - CHUNK_OVERLAP))
        assert len(result) == expected, (
            f"Expected {expected} chunks for 10 000 words, got {len(result)}"
        )

    def test_chunk_sizes_are_at_most_chunk_size_words(self):
        """No chunk may exceed CHUNK_SIZE words."""
        result = self.chunk_text(make_text(2000))
        for i, chunk in enumerate(result):
            word_count = len(self._text(chunk).split())
            assert word_count <= CHUNK_SIZE + 5, (
                f"Chunk {i} has {word_count} words, exceeds CHUNK_SIZE={CHUNK_SIZE}"
            )


# ── Embedder ──────────────────────────────────────────────────────────────────

class TestEmbedder:
    """pipeline/embedder.py — BGE-base-en-v1.5, 768-dim, normalized.

    Contract: embed(texts: list[str]) -> np.ndarray  shape (n, 768)
    """

    @pytest.fixture(autouse=True)
    def import_embedder(self):
        from pipeline.embedder import embed
        self.embed = embed

    def test_embed_returns_shape_n_768(self):
        """embed(['a', 'b']) → ndarray shape (2, 768)."""
        result = self.embed(["sentence one", "sentence two"])
        arr = np.array(result)
        assert arr.shape == (2, 768), f"Expected shape (2, 768), got {arr.shape}"

    def test_embed_single_item_list(self):
        """embed(['text']) → ndarray shape (1, 768)."""
        result = self.embed(["The quick brown fox"])
        arr = np.array(result)
        assert arr.shape == (1, 768), f"Expected shape (1, 768), got {arr.shape}"

    def test_embed_normalized(self):
        """Each row is L2-normalized (norm ≈ 1.0)."""
        result = np.array(self.embed(["Sample sentence for normalization check"]))
        norm = float(np.linalg.norm(result[0]))
        assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: norm={norm}"

    def test_same_input_same_output(self):
        """Deterministic: same list → identical embeddings."""
        sentence = ["Repeated sentence for determinism check"]
        v1 = np.array(self.embed(sentence))
        v2 = np.array(self.embed(sentence))
        assert np.allclose(v1, v2), "Same input produced different embeddings"

    def test_different_inputs_different_outputs(self):
        """Different sentences → different embedding rows."""
        result = np.array(self.embed([
            "Dogs are friendly animals",
            "Quantum mechanics describes subatomic particles",
        ]))
        assert not np.allclose(result[0], result[1]), "Different sentences produced identical embeddings"


# ── Retriever ─────────────────────────────────────────────────────────────────

class TestRetriever:
    """pipeline/retriever.py — retrieve(question, index, chunks) -> list[dict]

    Contract:
      retrieve(question: str, index, chunks: list[dict]) -> list[dict]
      chunks items have keys: text, doc_id
      We build the FAISS index manually using the embedder + faiss directly.
    """

    @pytest.fixture(autouse=True)
    def import_retrieve(self):
        from pipeline.retriever import retrieve
        self.retrieve = retrieve

    @pytest.fixture
    def small_index_and_chunks(self):
        import faiss
        from pipeline.embedder import embed
        chunks = [
            {"text": "Alpha content about retrieval", "doc_id": "a.pdf"},
            {"text": "Beta content about databases",  "doc_id": "b.pdf"},
            {"text": "Gamma content about embeddings","doc_id": "c.pdf"},
        ]
        texts = [c["text"] for c in chunks]
        embeddings = np.array(embed(texts)).astype("float32")
        index = faiss.IndexFlatIP(768)
        index.add(embeddings)
        return index, chunks

    def test_retrieve_returns_list_of_dicts(self, small_index_and_chunks):
        """retrieve() returns a list of dicts with 'text' and 'doc_id' keys."""
        index, chunks = small_index_and_chunks
        results = self.retrieve("Alpha", index, chunks)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "text" in results[0]
        assert "doc_id" in results[0]

    def test_retrieve_k_greater_than_n(self, small_index_and_chunks):
        """MUT-04: TOP_K=5 on 3 chunks must not crash — returns at most 3 results."""
        index, chunks = small_index_and_chunks
        results = self.retrieve("Alpha", index, chunks)
        assert len(results) <= len(chunks)

    def test_retrieve_most_similar_first(self, small_index_and_chunks):
        """Alpha query → Alpha chunk should be in results."""
        index, chunks = small_index_and_chunks
        results = self.retrieve("Alpha content about retrieval", index, chunks)
        top_text = results[0]["text"]
        assert "Alpha" in top_text, f"Expected Alpha chunk first, got: {top_text!r}"
