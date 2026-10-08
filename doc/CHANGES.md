# Code Changes

## Assignment-critical items (1–12) — status overview

| # | Topic | Status | Notes |
|---|---|---|---|
| 1 | Docker | Done | Dockerfile + docker-compose.yml created; image pre-downloads BGE model and EasyOCR weights |
| 2 | README | Done | Setup instructions, API examples, approach, development process narrative |
| 3 | Multi-file upload | Dropped | Reverted due to retrieval imbalance: large doc dominates FAISS index. Single-file kept. |
| 4 | OpenRouter timeout | Done | `timeout=(5, 60)` added to `requests.post` in `llm/backend.py` |
| 5 | HTTP status check | Done | `response.raise_for_status()` added before `.json()` in `llm/backend.py` |
| 6 | DistilBERT removed | Done | DistilBERT fallback removed entirely; `llm/backend.py` now requires OpenRouter — raises `RuntimeError` if key or model missing |
| 7 | PyMuPDF file handle leak | Done | `with pymupdf.open(...) as doc:` — context manager closes handle on every path (`extractor.py:37`) |
| 8 | Missing separator between page text and OCR | Done | `"\n"` added between body text and image text (`extractor.py:49`) |
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
| 16 | `_ocr_from_bytes` catches only `OSError` | Done | Now catches all exceptions — covered under #26 below |
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

**Note:** requests made through the Streamlit UI all reach the API from the frontend process,
so UI users share one rate-limit bucket. Direct API clients are limited per IP.

---

## #20 — Chunker infinite loop — skipped

**Problem:** Step is `CHUNK_SIZE - CHUNK_OVERLAP`. If `CHUNK_OVERLAP >= CHUNK_SIZE`, step ≤ 0
→ infinite loop.

**Why skipped:** `CHUNK_SIZE` and `CHUNK_OVERLAP` are system-level configuration values set by
whoever deploys the application — not end users. Anyone setting `CHUNK_OVERLAP >= CHUNK_SIZE`
has misconfigured the system and will observe the hang immediately on first upload. The default
values (`CHUNK_SIZE=350`, `CHUNK_OVERLAP=50`) are safe. Adding an assert is low effort but the
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
**Files:** `pipeline/embedder.py:11-17`, `ingestion/extractor.py:15-21`

(`llm/backend.py` also had a lazy loader at the time, for DistilBERT. It was removed together
with the DistilBERT fallback.)

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

Applied the same pattern to `_ocr_reader` in `extractor.py`.

---

## #23 — Relevance threshold in retriever
**Files:** `pipeline/retriever.py:17-19`, `config.py`, `.env`, `.env.example`

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

## #24 — Citation clause — reverted
**File:** `prompts/qa_prompt.txt`

**Original fix:** Removed "state which document(s) you used" from the hardcoded prompt as misleading when only one document exists.

**Reverted:** When the prompt was extracted to `prompts/qa_prompt.txt`, the citation instruction was restored as `"Cite which document (the bracketed filename) your answer comes from."` — now that context is labeled per source file (`[filename.pdf]`), citation is accurate and useful.

---

## #25 — `API_URL` now read from config in Streamlit frontend
**File:** `app.py`

**Problem:** `API_URL = "http://localhost:8000"` was hardcoded in `app.py` even though
`config.py` already defines `API_URL` as an env-var-backed value.

**Fix:** `from config import API_URL` — one source of truth.

---

## #26 — Consistent error handling in extractor
**File:** `ingestion/extractor.py:31-32, 54-55`

**Problem:** `extract_image` caught all exceptions and raised `ValueError`, but `extract_pdf`
caught nothing — any PyMuPDF failure would bubble up as a raw exception.
Also `_ocr_from_bytes` only caught `OSError`, letting other image errors escape.

**Fix:**
- `extract_pdf` now wrapped in `try/except`, raises `ValueError` on failure.
- `_ocr_from_bytes` now catches all exceptions (returns empty string on any failure).
- Both functions raise `ValueError` — consistent policy throughout.

---

## #27 — Magic-byte routing fixed (JPEG with .pdf extension)
**File:** `ingestion/extractor.py:72-80`

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
**File:** `api/app.py:34`

**Problem:** `logging.basicConfig` is a no-op if the root logger already has handlers attached
(uvicorn attaches its own handlers before app code runs in some configurations).
Our JSON handlers would silently not apply.

**Fix:** Added `force=True` (Python 3.8+). This removes any existing root handlers before
applying ours, guaranteeing our JSON formatter is always active.

---

## #30 — Updated `import pymupdf as fitz` → `import pymupdf`
**File:** `ingestion/extractor.py:3`

**Problem:** `import pymupdf as fitz` is a legacy alias from when the package was called
`fitz`. Modern PyMuPDF ships as `pymupdf`; the alias works but signals old code.

**Fix:** `import pymupdf` and updated all `fitz.open(...)` calls to `pymupdf.open(...)`.

---

## #11 — Max file size check
**Files:** `api/app.py:67-71`, `config.py`, `.env`, `.env.example`

**Problem:** No upper bound on uploaded file size. A large file would be read entirely into memory
before any rejection, potentially exhausting server resources.

**Fix:** Added `MAX_FILE_SIZE_MB` config value (default `50`). After reading contents into memory,
the endpoint checks `len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024` and returns 413 before
writing anything to disk.

---

## #12 — Path traversal in temp file path
**File:** `api/app.py:73-76`

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

## MUT02 / MUT03 — Wrong file type routing no longer crashes server
**File:** `ingestion/extractor.py`

**Problem:** In the first version, routing was by extension only. When PDF bytes were sent with a
`.png` extension (or wrong Content-Type), the server routed them to EasyOCR. EasyOCR cannot handle
PDF bytes and the server crashed with a 500 or dropped the connection.

**Fix:** Magic bytes are checked FIRST (see #27). If the file starts with `%PDF`, it goes to
PyMuPDF regardless of extension or Content-Type. A PNG with a `.pdf` extension fails `_is_pdf()`,
has no image extension, and is rejected with a clean 422. No crash in either direction.

---

## TC01–TC02 — Failed upload does not wipe existing session index
**File:** `api/app.py:101`

**Risk:** If a second upload to the same session fails (empty file, unsupported type,
extraction error) and the server saves an empty or partial result, the existing session index
is wiped and a subsequent `/ask` cannot answer from the previously indexed content.

**Guarantee:** `store.save()` is only called at the very end of the upload handler, after all
validation and extraction succeed. Any failure raises `HTTPException` before `store.save()`
is reached — the existing session index is never touched. TC01 and TC02 verify this: the
session survives a failed upload. TC03–TC07 cover other upload/ask sequences.

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
`requests.HTTPError` with the status code, which the API returns as a 502 with a useful message
(see "LLM error codes" under later changes).

### #9 — `OPENROUTER_MODEL` now read per-request
**File:** `llm/backend.py`

**Problem:** `OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")` was at module level — read once
at import time. If the env var was set after import, the change was invisible until restart.
Also, if key is set but model isn't, `"model": None` would be sent to OpenRouter.

**Fix:** Both `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` are now read inside `answer()` per
request. Raises a clear `RuntimeError` if either is missing.

---

## #1 — Docker
**Files:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`

Two-service Docker setup: `api` (FastAPI, port 8000) and `frontend` (Streamlit, port 8501).

- Base image: `python:3.11-slim` + system libs for EasyOCR (`libgl1`, `libglib2.0-0`)
- BGE embedding model (`BAAI/bge-base-en-v1.5`) pre-downloaded during `docker build` — avoids slow first-request download at runtime
- EasyOCR weights pre-downloaded during `docker build` as well (added later)
- `.env` passed at runtime via `env_file` — secrets never baked into the image
- `API_URL=http://api:8000` injected into `frontend` service so Streamlit reaches the API by Docker service name, not localhost
- `.dockerignore` excludes `.env`, `.venv/`, `__pycache__`, `.git/`, `.idea/`

Build and run: `docker-compose up --build`

---

## Additional changes (post-review round)

### DistilBERT removed
**File:** `llm/backend.py`, `requirements.txt`

DistilBERT fallback removed entirely. `torch` and `transformers` removed from `requirements.txt`. `llm/backend.py` now only calls OpenRouter — raises `RuntimeError` with a clear message if `OPENROUTER_API_KEY` or `OPENROUTER_MODEL` is not set.

### Prompt extracted to file
**File:** `prompts/qa_prompt.txt`

LLM prompt moved from hardcoded string in `llm/backend.py` to `prompts/qa_prompt.txt`. Loaded at import time with a `FileNotFoundError` guard. Prompt instructs the model to: answer only from provided excerpts, cite the source document by bracketed filename, explain its reasoning, and say explicitly if the answer is not in the excerpts.

### Log file path fixed
**File:** `api/app.py:30-31`, `.env.example`

`app.log` was written to the current working directory, which is not writable in Docker. Now defaults to `tempfile.gettempdir()/app.log` (always writable). Overridable via `LOG_FILE` env var.

### Streamlit upload limit aligned
**File:** `.streamlit/config.toml`

Streamlit's default upload limit is 200 MB. Created `.streamlit/config.toml` with `maxUploadSize = 50` to match the API's `MAX_FILE_SIZE_MB = 50`. **Note:** if `MAX_FILE_SIZE_MB` is changed in `.env`, `.streamlit/config.toml` must be updated manually to stay in sync.

### Test suite README renamed
**File:** `tests/README.md` → `tests/TESTING.md`

Renamed to `TESTING.md` to match the naming convention of other doc files (`SPEC.md`, `CHANGES.md`). Content unchanged.

### Test documents added
**Directory:** `test_docs/`

Sample documents added to the repository as required by the assignment ("find or create random/dummy test docs, store in GitHub repo"). Includes PDFs, PNG/JPG invoices and contracts, a BMP stock table, and a handwriting sample — covering all supported file types and real-world document categories.

---

## Later changes

### Minimum file size guard
**File:** `api/app.py:64-65`

Uploads smaller than 100 bytes are rejected with 422 before the file reaches PyMuPDF or EasyOCR.

### Handlers run on the threadpool
**File:** `api/app.py`

`upload` and `ask` changed from `async def` to `def`, so FastAPI runs them on its threadpool.
Previously the synchronous work inside `async def` handlers was serialized on the event loop.

### EasyOCR warm-up and pre-download
**Files:** `api/app.py:43-44`, `Dockerfile`

The EasyOCR reader is warmed up in `lifespan` alongside the embedder, so the first image upload
does not pay the cold-start cost. The Docker image pre-downloads the EasyOCR weights at build time.

### Frontend waits for the API
**File:** `app.py`

On first load the Streamlit app polls the API's `/openapi.json` and shows a spinner until the
API is ready. Prevents "connection refused" when Upload is clicked during model warm-up.

### LLM error codes
**File:** `api/app.py:125-134` (the `try/except` around `backend.answer`)

Upstream OpenRouter failures are mapped to specific HTTP codes: 504 on timeout, 502 on
connection or HTTP error, 500 on missing configuration. Previously all were a generic 500.

### Default chunk size 500 → 350
**Files:** `config.py`, `.env.example`

BGE-base accepts at most 512 tokens, and a 500-word chunk is expected to exceed that, so the end
of each chunk would be cut off before embedding. 350 words is an estimate of what fits; it was
not measured.

### Corrupt PDF with a valid header returned 500 on Windows
**File:** `api/app.py:78-87`

**Problem:** When PyMuPDF failed to open a file that starts with `%PDF`, the handler removed the
temp file inside the `except` block. On Windows the file was still held open at that point, so
`os.remove` raised `PermissionError`, the request ended in a 500, and the temp file stayed on disk.

**Fix:** The handler stores the error message, leaves the `except` block, removes the temp file,
and only then raises the 422. Found by MUT04 once its payload was made larger than the size guard.

### Unsupported-type tests reach the type check
**Files:** `tests/test_upload.py`, `tests/test_integration.py`

T004, T005, T006, TC02, MUT04 and the supported-vs-unsupported test sent payloads under 100 bytes,
so the size guard rejected them before the type check ran. The payloads are now larger than the guard.

### Test suite cleanup
**Files:** `tests/`, `session/store.py`

- Added T014: a file just over `MAX_FILE_SIZE_MB` returns 413. The size limit had no test.
- T039b test function renamed to match its ID (it was named `test_T039_...`).
- T050 expected count now uses the chunker's own formula, `ceil(n / step)`.
- Docstrings and test docs for T028, T030, T037, TC03, TC07, MUT01 and MUT02 now state what the tests assert.
- Removed four unused fixtures from `conftest.py` and unused settings from `pytest.ini`.
- Mutation tests renumbered from CAT-05 to CAT-12; CAT-05 was used twice.
- `store.save` type hint corrected: `chunks: list[dict]`.
- Tests without IDs are now listed in `tests/TESTING.md` and `tests/test_spec.csv`.
- T004, T006, T017, T036 and the supported-vs-unsupported test asserted `>= 400`, which a 500 would pass. They now assert a 4xx.
- T035 asserts non-empty answers; the chunk-size test asserts `<= CHUNK_SIZE` exactly (it allowed 5 extra words).
- Unused `UPLOAD_PROBE` removed from the rate limit tests; the `/upload` rate limit has no test.
