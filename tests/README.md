# Document QA — Test Suite

Black-box API tests for the Document QA backend. All tests hit a live server over HTTP — no mocking, no test client.

## Prerequisites

1. **Server running** at `http://localhost:8000`
2. **Dependencies installed** in the project venv:
   ```
   pip install pytest httpx pillow
   ```
   PyMuPDF and FAISS are already in the project's main dependencies.

## Running the tests

From the **project root** (not the `tests/` directory):

```bash
.venv\Scripts\python.exe -m pytest tests\ -v
```

## Test structure

| ID | File | What it tests |
|----|------|---------------|
| T001–T003 | `test_upload.py` | Supported file types (PDF, PNG, JPG) accepted → 200 |
| T004–T006 | `test_upload.py` | Unsupported types (exe, docx, csv) rejected → 4xx |
| T007–T010 | `test_upload.py` | File count — no file, one file, sequential uploads |
| T011–T013 | `test_upload.py` | File state — empty, corrupt, normal size |
| T016–T020 | `test_upload.py` | Session UUID — missing, empty, valid, malformed, re-upload |
| T021–T023 | `test_ask.py` | Pre-conditions — ask before upload (404), no session_id, after upload |
| T024–T030 | `test_ask.py` | Question content — empty, whitespace, normal, very long, off-topic, injection |
| T031–T038 | `test_integration.py` | End-to-end flows — PDF, image, session isolation, concurrency, fallback |
| TC01 | `test_integration.py` | Valid upload → empty file attempt → ask still works (index not wiped) |
| TC02 | `test_integration.py` | Valid upload → unsupported file attempt → index survives rejection |
| TC03 | `test_integration.py` | Valid upload → re-upload → ask reflects updated content |
| TC04 | `test_integration.py` | Image (OCR) upload → PDF upload → ask works across both |
| TC05 | `test_integration.py` | Failed upload → recovery upload → session works normally |
| TC06 | `test_integration.py` | Ask (404 on empty session) → upload → ask works (no broken state) |
| TC07 | `test_integration.py` | Mixed-type session → off-topic question → LLM says not found |
| T039–T041 | `test_rate_limit.py` | slowapi rate limiting — normal, burst (429), cooldown recovery |
| T042–T045 | `test_pipeline_units.py` | Extractor — PDF text layer, blank PDF, image OCR |
| T046–T050 | `test_pipeline_units.py` | Chunker — empty, boundary, overlap, large text |
| — | `test_pipeline_units.py` | Embedder — shape (768-dim), normalized, deterministic |
| — | `test_pipeline_units.py` | Retriever — returns dicts, top-k, similarity order |
| T046–T048 | `test_store.py` | Session store — save/load, unknown UUID, overwrite, isolation |

## Notes

- **Rate limit tests** (T040, T041) skip automatically if the server has no rate limit configured. Once slowapi is active they run without any changes.
- **DistilBERT fallback test** (T038) requires starting the server without `OPENROUTER_API_KEY`.
- Tests run against whatever server is at `localhost:8000` — start it before running pytest.
