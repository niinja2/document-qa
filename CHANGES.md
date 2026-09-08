# Code Changes

## Assignment-critical items (1–12) — status overview

| # | Topic | Status | Notes |
|---|---|---|---|
| 1 | Docker | Pending | Dockerfile + docker-compose.yml — not yet created |
| 2 | README | Pending | Setup instructions, examples, approach description — not yet created |
| 3 | Multi-file upload | Dropped | Reverted due to retrieval imbalance: large doc dominates FAISS index. Single-file kept. |
| 4 | OpenRouter timeout | Done | `timeout=(5, 60)` added to `requests.post` in `llm/backend.py` |
| 5 | HTTP status check | Done | `response.raise_for_status()` added before `.json()` in `llm/backend.py` |
| 6 | DistilBERT context truncation | Dropped | DistilBERT to be replaced; tokenizer already truncates at 512 tokens |
| 7 | — | Not actioned | Not flagged as assignment-critical |
| 8 | — | Not actioned | Not flagged as assignment-critical |
| 9 | `OPENROUTER_MODEL` per-request | Done | Both key and model now read inside `answer()` per request |
| 10 | UUID regenerated on every click | Dropped | Intentional: each upload starts a fresh session |
| 11 | Max file size check | Done | `MAX_FILE_SIZE_MB` config added; 413 returned before writing to disk |
| 12 | Path traversal in temp path | Done | `tempfile.NamedTemporaryFile` used; user input never in file path |

---

## "Fix soon" items (5, 7, 8, 9, 14, 16, 20, 21) — status overview

| # | Topic | Status | Notes |
|---|---|---|---|
| 5 | HTTP status check on OpenRouter | Done | `response.raise_for_status()` added — see detail below |
| 7 | PyMuPDF file handle leak | Done | `with pymupdf.open(...) as doc:` — see cleanup round |
| 8 | Missing separator page text + OCR | Done | `"\n"` added between body and image text — see cleanup round |
| 9 | `OPENROUTER_MODEL` read at import time | Done | Both key and model now read per-request — see detail below |
| 14 | `get_remote_address` collapses behind proxy | Skipped | Not relevant for this deployment — see reasoning below |
| 16 | `_ocr_from_bytes` catches only `OSError` | Done | Now catches all exceptions — see cleanup round |
| 20 | Chunker infinite loop if `CHUNK_OVERLAP >= CHUNK_SIZE` | Skipped | Not relevant for this use case — see reasoning below |
| 21 | Session store has no TTL or eviction | Skipped | Not relevant at demo scale — see reasoning below |

---

## #14 — `get_remote_address` behind a proxy — skipped

**Problem:** slowapi uses the raw socket IP for rate limiting. Behind nginx or a load balancer,
every request arrives from `127.0.0.1` (the proxy), not the real client. All users share one
rate-limit bucket, making per-user limiting ineffective.

**Why skipped:** This project runs locally — Streamlit and the API on the same machine, no proxy
in between. `get_remote_address` reads the correct IP and rate limiting works as intended.
The issue only manifests in a production deployment behind a reverse proxy. For a local demo
this is a non-issue.

---

## #20 — Chunker infinite loop — skipped

**Problem:** Step is `CHUNK_SIZE - CHUNK_OVERLAP`. If `CHUNK_OVERLAP >= CHUNK_SIZE`, step ≤ 0
→ infinite loop.

**Why skipped:** `CHUNK_SIZE` and `CHUNK_OVERLAP` are system-level configuration values set by
whoever deploys the application — not end users. Anyone setting `CHUNK_OVERLAP >= CHUNK_SIZE`
has misconfigured the system and will observe the hang immediately on first upload. The default
values (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`) are safe. Adding an assert is low effort but the
failure mode is obvious and self-correcting for the operator.

---

## #21 — Session store no TTL / no eviction — skipped

**Problem:** Every upload adds a FAISS index to an in-memory dict. Nothing is ever removed.
Python's garbage collector will not free these because the dict holds an explicit reference.
RAM grows for the life of the process.

**Why skipped:** This matters in a long-running production system with many users uploading
continuously. For a demo with a handful of documents, the memory footprint is negligible and
the process restarts between sessions. An LRU cap would be the right fix for a production
system, but it is out of scope for this assignment.

---

## Cleanup Round (items 17–30 from critical review)

## #19 — Deleted `test_ocr.py` from project root
Stray debug script, not part of the test suite, not tracked in `tests/`. Deleted.

---

## #22 — Thread-safe lazy loaders
**Files:** `pipeline/embedder.py`, `ingestion/extractor.py`, `llm/backend.py`

**Problem:** The lazy-loader pattern `if _model is None: _model = load()` is not thread-safe.
Two concurrent requests hitting the server for the first time could both pass the `None` check
simultaneously and load the model twice, causing a race condition.

**Fix:** Double-checked locking — check once outside the lock (fast path), lock, check again inside
(safe path), then load. Only the first thread does the work; all others wait and reuse.

```python
_lock = threading.Lock()

def _get_model():
    global _model
    if _model is None:          # fast check (no lock overhead once loaded)
        with _lock:
            if _model is None:  # safe check (only one thread loads)
                _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model
```

Applied the same pattern to `_ocr_reader` in `extractor.py` and `_qa_model` in `backend.py`.

---

## #23 — Relevance threshold in retriever
**Files:** `pipeline/retriever.py`, `config.py`, `.env`, `.env.example`

**Problem:** `retrieve()` always returned top-K chunks regardless of how relevant they were.
For an off-topic question, the LLM would receive 5 unrelated chunks and might hallucinate.

**Fix:** Filter chunks by cosine similarity score. FAISS `IndexFlatIP` on normalized vectors
returns cosine similarity (range -1 to 1). Added `RETRIEVAL_THRESHOLD` config value (default `0.0`).
At default, behaviour is unchanged — only FAISS padding slots (score = -1, index = -1) are dropped,
which were already filtered. Raise the threshold in `.env` to get stricter filtering.

```python
for score, i in zip(distances[0], indices[0]):
    if i >= 0 and score >= RETRIEVAL_THRESHOLD:
        hit_positions.append(i)
```

---

## #24 — Removed fake citation clause from LLM prompt
**File:** `llm/backend.py`

**Problem:** The prompt said "state which document(s) you used" — but only one document exists
per session. The model would always cite the one filename regardless of whether it actually
contributed to the answer. Misleading.

**Fix:** Removed the sentence. The prompt still asks the model to answer only from the excerpts
and say so explicitly if the answer isn't there.

---

## #25 — `API_URL` now read from config in Streamlit frontend
**File:** `app.py`

**Problem:** `API_URL = "http://localhost:8000"` was hardcoded in `app.py` even though
`config.py` already defines `API_URL` as an env-var-backed value.

**Fix:** `from config import API_URL` — one source of truth.

---

## #26 — Consistent error handling in extractor
**File:** `ingestion/extractor.py`

**Problem:** `extract_image` caught all exceptions and raised `ValueError`, but `extract_pdf`
caught nothing — any PyMuPDF failure would bubble up as a raw exception.
Also `_ocr_from_bytes` only caught `OSError`, letting other image errors escape.

**Fix:**
- `extract_pdf` now wrapped in `try/except`, raises `ValueError` on failure.
- `_ocr_from_bytes` now catches all exceptions (returns empty string on any failure).
- Both functions raise `ValueError` — consistent policy throughout.

---

## #27 — Magic-byte routing fixed (JPEG with .pdf extension)
**File:** `ingestion/extractor.py`

**Problem:** Old routing: `if path.endswith(".pdf") or _is_pdf(file_path)`.
A JPEG with a `.pdf` extension matched `endswith(".pdf")` first and was sent to PyMuPDF,
which would error. Only PDF bytes with a non-pdf extension were correctly caught by `_is_pdf`.

**Fix:** Check magic bytes FIRST, then fall back to extension:

```python
if _is_pdf(file_path):                              # actual PDF bytes → PyMuPDF
    text = extract_pdf(file_path)
elif path.endswith((".png", ".jpg", ".jpeg", ...)):  # image extension → EasyOCR
    text = extract_image(file_path)
else:
    raise ValueError(f"Unsupported file type: {file_path}")
```

A JPEG with `.pdf` extension now fails `_is_pdf()`, fails the image extension check, and gets
a clean `ValueError` → 422. A PDF with `.png` extension correctly goes to `extract_pdf`.

---

## #28 — Structured error contract
**Skipped.** Would require a full error-mapping layer across all endpoints. Out of scope
for this cleanup round.

---

## #29 — `logging.basicConfig` fixed with `force=True`
**File:** `api/app.py`

**Problem:** `logging.basicConfig` is a no-op if the root logger already has handlers attached
(uvicorn attaches its own handlers before app code runs in some configurations).
Our JSON handlers would silently not apply.

**Fix:** Added `force=True` (Python 3.8+). This removes any existing root handlers before
applying ours, guaranteeing our JSON formatter is always active.

---

## #30 — Updated `import pymupdf as fitz` → `import pymupdf`
**File:** `ingestion/extractor.py`

**Problem:** `import pymupdf as fitz` is a legacy alias from when the package was called
`fitz`. Modern PyMuPDF ships as `pymupdf`; the alias works but signals old code.

**Fix:** `import pymupdf` and updated all `fitz.open(...)` calls to `pymupdf.open(...)`.

---

## #11 — Max file size check
**Files:** `config.py`, `api/app.py`, `.env`, `.env.example`

**Problem:** No upper bound on uploaded file size. A large file would be read entirely into memory
before any rejection, potentially exhausting server resources.

**Fix:** Added `MAX_FILE_SIZE_MB` config value (default `50`). After reading contents into memory,
the endpoint checks `len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024` and returns 413 before
writing anything to disk.

---

## #12 — Path traversal in temp file path
**File:** `api/app.py`

**Problem:** `session_id` is a user-supplied form field. The old code used it to build the temp
file path:

```python
tmp_path = str(Path(tempfile.gettempdir()) / f"{session_id}{suffix}")
```

An attacker calling the API directly could send `session_id=../../windows/system32/foo`, causing
the server to write uploaded bytes to an arbitrary location on the filesystem.

**Fix:** Replaced with `tempfile.NamedTemporaryFile(suffix=suffix, delete=False)`. The OS picks
the path; user input never touches the filesystem path.

---

## Bonus fixes applied alongside (from "fix soon" list)

### #4 — OpenRouter request timeout
**File:** `llm/backend.py`

Added `timeout=(5, 60)` to the `requests.post` call — 5s to connect, 60s to receive.
Without this, a hung OpenRouter upstream would block a FastAPI worker indefinitely.

### #5 — HTTP status check on OpenRouter response
**File:** `llm/backend.py`

Added `response.raise_for_status()` before `.json()`. A 5xx HTML response or a 4xx error
would previously cause a `JSONDecodeError` surfacing as a confusing 500. Now it raises
`requests.HTTPError` with the status code, which bubbles up as a clean 500 with a useful message.

### #9 — `OPENROUTER_MODEL` now read per-request
**File:** `llm/backend.py`

**Problem:** `OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")` was at module level — read once
at import time. If the env var was set after import, the change was invisible until restart.
Also, if key is set but model isn't, `"model": None` would be sent to OpenRouter.

**Fix:** Both `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` are now read inside `answer()` per
request. Falls back to DistilBERT if either is missing.
