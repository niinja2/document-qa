# Document QA — Test Suite

Black-box API tests for the Document QA backend. All tests hit a live server over HTTP — no mocking, no test client.

## Prerequisites

1. **Server running** at `http://localhost:8000`
2. **Dependencies installed** in the project venv:
   ```
   pip install pytest httpx pillow
   ```
   PyMuPDF and FAISS are already in the project's main dependencies.

## How to run

There are two separate runs because the main suite and the rate limit tests require different server configurations.

---

### Run 1 — Main suite (high rate limits)

Start the server with raised limits so the full suite doesn't get throttled:

```powershell
$env:RATE_LIMIT_UPLOAD = "1000/minute"; $env:RATE_LIMIT_ASK = "1000/minute"
.venv\Scripts\uvicorn.exe api.app:app --host 0.0.0.0 --port 8000
```

Then in a second terminal, from the project root:

```bash
.venv\Scripts\python.exe -m pytest tests\ --ignore=tests\test_z_rate_limit.py -v
```

---

### Run 2 — Rate limit tests (default limits)

Stop the server. Clear the env vars and restart with default limits from `.env`:

```powershell
Remove-Item Env:RATE_LIMIT_UPLOAD; Remove-Item Env:RATE_LIMIT_ASK
.venv\Scripts\uvicorn.exe api.app:app --host 0.0.0.0 --port 8000
```

Then run only the rate limit file:

```bash
.venv\Scripts\python.exe -m pytest tests\test_z_rate_limit.py -v
```

> **Note:** T041 waits ~62 seconds for the rate limit window to reset — this is expected. Total run time is ~90 seconds.

---

## Test structure

| ID | File | What it tests |
|----|------|---------------|
| T001–T003 | `test_upload.py` | Supported file types (PDF, PNG, JPG) accepted → 200 |
| T004–T006 | `test_upload.py` | Unsupported types (exe, docx, csv) rejected → 4xx |
| T007–T010 | `test_upload.py` | File count — no file, one file, sequential uploads |
| T011–T013 | `test_upload.py` | File state — empty, corrupt, normal size |
| T016–T020 | `test_upload.py` | Session UUID — missing, empty, valid, malformed, re-upload |
| MUT01–MUT12 | `test_upload.py` | Mutation tests — mismatched types, null bytes, duplicate fields |
| T021–T030 | `test_ask.py` | Ask pre-conditions and question content — 404, 422, injections, edge cases |
| T031–T038 | `test_integration.py` | End-to-end flows — PDF, image, session isolation, concurrency |
| T039b | `test_integration.py` | LLM response format — no leaked error keys in response JSON |
| TC01–TC07 | `test_integration.py` | Cascade tests — state interaction across multiple uploads and asks |
| T039–T041 | `test_z_rate_limit.py` | Rate limiting — normal request, burst triggers 429, cooldown recovery |
| T042–T050 | `test_pipeline_units.py` | Unit tests — extractor, chunker, embedder, retriever |
| TS01–TS03 | `test_store.py` | Session store — save/load, unknown UUID, overwrite |

## Full test list

### /upload — File Type (T001–T006)

| ID | Description |
|----|-------------|
| T001 | PDF with selectable text → 200, text extracted |
| T002 | PNG with printed text → 200, OCR applied |
| T003 | JPG with printed text → 200, OCR applied |
| T004 | Unsupported .exe binary → 4xx |
| T005 | Unsupported .docx → 422 |
| T006 | Unsupported .csv → 4xx |

### /upload — File Count (T007–T010)

| ID | Description |
|----|-------------|
| T007 | No file field in multipart → 422 |
| T008 | Single file upload → 200 |
| T009 | Two sequential uploads to same session → both 200 |
| T010 | Five sequential uploads to same session → all 200, no crash |

### /upload — File Size (T011–T013)

| ID | Description |
|----|-------------|
| T011 | Empty file (0 bytes) → 422, size guard fires before PyMuPDF |
| T012 | Corrupt 1-byte file → 422, size guard fires before PyMuPDF |
| T013 | Normal size PDF → 200 |

### /upload — Session UUID (T016–T020)

| ID | Description |
|----|-------------|
| T016 | No session_id field → 422 |
| T017 | Empty string session_id → 4xx |
| T018 | Well-formed UUID v4 → 200 |
| T019 | Malformed session_id string → not 500 (behaviour unspecified) |
| T020 | Re-upload to same UUID twice → 200 both times, no 409 |

### /ask — Pre-conditions & Question Content (T021–T030)

| ID | Description |
|----|-------------|
| T021 | Ask before any upload → 404 session not found |
| T022 | No session_id field → 422 |
| T023 | Ask after successful upload → 200, non-empty answer |
| T024 | Empty string question → 4xx or graceful, not 500 |
| T025 | Whitespace-only question → 4xx or graceful, not 500 |
| T026 | Normal question about document content → 200, non-empty answer |
| T027 | Very long question (>2000 chars) → 200 or 4xx, not 500 |
| T028 | Question about content not in document → 200, LLM says not found |
| T029 | SQL injection in question → 200, treated as plain text |
| T030 | XSS payload in question → 200, returned as escaped JSON |

### Integration — End-to-End (T031–T038)

| ID | Description |
|----|-------------|
| T031 | PDF upload then ask → answer references PDF content |
| T032 | PNG upload → OCR → ask → answer reflects OCR'd text |
| T033 | PDF + PNG uploads then ask → both succeed, second replaces first, ask returns answer |
| T034 | Two sessions — session A cannot see session B's content |
| T035 | Three sequential questions same session → all 200, no state bleed |
| T036 | Ask with unknown UUID → 4xx (simulates post-restart state) |
| T037 | Two concurrent uploads different sessions → both 200, no data mixing |
| T038 | No OpenRouter key → 500 error (DistilBERT fallback removed) |

### Cascade — State Interaction (TC01–TC07)

| ID | Description |
|----|-------------|
| TC01 | Valid upload → empty upload (422) → ask → index not wiped |
| TC02 | Valid upload → unsupported upload (422) → ask → index survives |
| TC03 | Valid upload → re-upload different PDF → ask → index updated |
| TC04 | PNG upload → PDF upload → both succeed → ask returns answer |
| TC05 | Failed upload → recovery valid upload → ask → session recovers |
| TC06 | Ask on empty session (404) → upload → ask → no broken state |
| TC07 | PDF + PNG session → off-topic question → LLM says not found |

### Mutation — Single-Property Changes (MUT01–MUT12)

| ID | Description |
|----|-------------|
| MUT01 | PNG bytes sent with .pdf filename → not 500 |
| MUT02 | PDF bytes sent with .png filename → not 500 |
| MUT03 | Valid PDF bytes but Content-Type claims image/png → not 500 |
| MUT04 | Valid %PDF header but truncated body → 422, graceful error |
| MUT05 | UUID with uppercase hex letters → not 500 |
| MUT06 | UUID with extra characters appended → not 500 |
| MUT07 | Duplicate session_id fields in multipart → not 500 |
| MUT08 | Null byte embedded in question → not 500 |
| MUT09 | Unicode control characters in question → not 500 |
| MUT10 | Right-to-left Unicode question (Arabic) → not 500 |
| MUT11 | Numbers and symbols only question → not 500 |
| MUT12 | Extra unexpected field in upload multipart → not 500 |

### Integration — LLM Response Format (T039b)

| ID | Description |
|----|-------------|
| T039b | Upload then ask → response JSON has `answer` key, no `traceback`/`exception` keys leaked |

### Rate Limit (T039–T041)

| ID | Description |
|----|-------------|
| T039 | Single /ask request → 200, not throttled |
| T040 | 35 sequential /ask requests → 429 triggered within window |
| T041 | Burst triggers 429, wait 62s cooldown → back to 200 |

### Unit — Extractor, Chunker, Store (T042–TS03)

| ID | Description |
|----|-------------|
| T042 | PDF text layer extraction → non-empty string |
| T043 | Blank PDF extraction → empty string or known exception, not crash |
| T044 | PNG image extraction via EasyOCR → non-empty string |
| T046 | Chunker: empty string → empty list |
| T047 | Chunker: 449 words → 1 chunk |
| T048 | Chunker: 450 words → 1 chunk |
| T049 | Chunker: 451 words → 2 chunks, second starts at word450 |
| T050 | Chunker: 10000 words → 22 chunks |
| TS01 | Store: save then load returns same data |
| TS02 | Store: load unknown UUID returns None, not KeyError |
| TS03 | Store: overwrite same UUID — second save replaces first |

## Notes

- **Rate limit tests** (`@pytest.mark.rate_limit`, file `test_z_rate_limit.py`) run against the server with **default limits** (`.env` values, no env var overrides). Run them separately as described above. The `z_` prefix forces alphabetical ordering so these tests run last and don't pollute the rate limit window for other tests. The file was renamed from `test_rate_limit.py` for this reason.
- **T038** (no OpenRouter key) — DistilBERT fallback has been removed. Running without `OPENROUTER_API_KEY` now returns 500. T038 as written only checks for 200 + non-empty answer, so it will fail in this configuration — skip it or update the assertion if testing the error path.
- Tests run against whatever server is at `localhost:8000` — start it before running pytest.
- **To skip rate limit tests** in the main suite run: `.venv\Scripts\python.exe -m pytest tests\ --ignore=tests\test_z_rate_limit.py -v` (already the recommended command above).
