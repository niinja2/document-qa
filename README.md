# Document QA

A REST API that extracts text from uploaded documents (PDF, images) and answers natural language questions about them using a RAG pipeline and an LLM.

Built with FastAPI + Streamlit frontend. Runs locally or via Docker.

---

## What it does

1. **Upload** a PDF or image → text is extracted, chunked, embedded, and indexed in FAISS
2. **Ask** a question → top relevant chunks are retrieved, sent to an LLM with source labels, answer is returned with citations and reasoning

---

## Setup

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai) API key and a model name (e.g. `mistralai/ministral-3b-2410`)

### Manual

```bash
git clone <repo-url>
cd document-qa
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your key:

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=mistralai/ministral-3b-2410
```

Start the API:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Start the frontend (separate terminal):

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

### Docker

```bash
docker-compose up --build
```

Open `http://localhost:8501`. The API runs on port 8000, Streamlit on 8501.

The `.env` file is read at runtime — it is never baked into the image.

---

## API reference

### POST /upload

Upload a document and index it under a session UUID.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@contract.pdf" \
  -F "session_id=550e8400-e29b-41d4-a716-446655440000"
```

Response:
```json
{"status": "ok"}
```

Error codes: `413` file too large, `422` unsupported type / empty file / extraction failed, `429` rate limit.

### POST /ask

Ask a question about the uploaded document.

```bash
curl -X POST http://localhost:8000/ask \
  -F "session_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "question=What is the total contract value?"
```

Response:
```json
{"answer": "The total contract value is $45,000, as stated in Section 3.2 of [contract.pdf]."}
```

Error codes: `404` session not found (upload first), `422` missing field, `429` rate limit, `500` LLM error.

---

## Supported file types

| Type | Extraction method |
|---|---|
| PDF (text layer) | PyMuPDF |
| PDF (scanned / embedded images) | PyMuPDF + EasyOCR |
| PNG, JPG, JPEG, TIFF, BMP | EasyOCR |

File routing uses magic bytes (`%PDF`), not the file extension — a PDF sent with a `.png` extension is handled correctly.

---

## How it works

### Scope decisions

The optional enhancements — NER, Redis, RAG — were evaluated against what the application actually needs to do. We considered multi-user support, multiple documents per user, large document sizes, and persistent storage. Supporting multiple users properly isn't just one feature — it's a whole system: authentication, per-user storage, session recovery across restarts, user-level logging. All of that is real engineering effort, and none of it is needed for a demo that handles one person uploading a document and asking questions about it. We didn't want to do three things badly instead of doing one thing well.

RAG was the one enhancement worth adding — not because the core requirement demanded it, but because it solves a real problem: large documents can't be sent to an LLM in one shot. Chunking and vector retrieval let us handle documents of any size and send only the relevant parts to the model. At demo scale with single-file sessions, the LLM could handle most documents directly. RAG was chosen because it's the right architectural direction — it handles arbitrarily large documents, and it's the foundation any production version of this system would build on. The complexity it introduces is discussed below.

Out of the general optional enhancements, we implemented a Streamlit UI, structured JSON logging, modular code organization, rate limiting, and input sanitization. Testing was added on top — not listed as an optional enhancement, but treated as a requirement because we wanted to guarantee the quality of the system before delivery. The testing approach is described in the development process section below.

### Architecture

Two-process system: FastAPI backend + Streamlit frontend. Both containerised via Docker Compose.

RAG pipeline:
- **Chunker** — 500-word chunks, 50-word overlap
- **Embedder** — `BAAI/bge-base-en-v1.5` (768-dim, L2-normalized), via `sentence-transformers`
- **Index** — FAISS `IndexFlatIP` (cosine similarity on normalized vectors)
- **Retriever** — top-5 chunks by cosine score, filtered by `RETRIEVAL_THRESHOLD`
- **LLM** — OpenRouter (model configurable via `.env`)

The architecture follows naturally from the two endpoints:

- `POST /upload`: extract text → chunk → embed → index → store in session
- `POST /ask`: embed question → retrieve top chunks → label by source → send to LLM → return answer

Text extraction routes by magic bytes — if the file starts with `%PDF` it goes to PyMuPDF (with EasyOCR for embedded images), everything else goes to EasyOCR directly. A PDF sent with a `.png` extension is handled correctly; a JPEG sent with a `.pdf` extension fails cleanly instead of crashing.

Sessions are stored in RAM, keyed by UUID. One FAISS index per session. No persistence across restarts.

### Why these choices

- **BGE over other embedding models** — strong retrieval benchmark performance, freely available, fits in RAM
- **FAISS `IndexFlatIP`** — exact cosine search, no approximation needed at demo scale
- **OpenRouter** — model-agnostic; swap the model via `.env` without code changes
- **Single file per session** — multi-file upload was implemented then reverted: large documents dominated the FAISS index, causing the retriever to ignore smaller documents entirely
- **Magic bytes routing** — more reliable than trusting file extensions or `Content-Type` headers

### LLM prompt

The prompt instructs the model to:
- Answer only from the provided excerpts (no hallucination)
- Cite the source document by filename
- Explain its reasoning
- Say explicitly if the answer is not in the excerpts

Prompt is in `prompts/qa_prompt.txt`.

### What we tried and abandoned

Multi-file upload was implemented first. The idea was to index all uploaded documents together in one FAISS index and retrieve across all of them. The problem: add multi-file upload, and you immediately need a way to make sure the retriever doesn't ignore smaller documents in favour of larger ones. Fix that, and you need more sophisticated index management. One decision pulls ten more behind it. It was reverted. Single file per session, index replaced on each upload. (`doc/CHANGES.md` #3)

DistilBERT was initially included as a local fallback — run a QA model on-device when no OpenRouter key is configured. It was removed for two reasons: its 512-token context limit meant it would silently truncate most real documents and degrade quality without warning, and OpenRouter already covers the use case better. A clear error saying "set your API key" is better than a fallback that quietly produces worse answers. (`doc/CHANGES.md` #6)

The prompt started as a hardcoded string inside `backend.py`. LLM instructions are not code — they belong in a file where they can be read, edited, and versioned independently. Moving it to `prompts/qa_prompt.txt` also made it easier to improve: anti-hallucination instructions, source citation, and reasoning explanation were added. The citation clause has its own small history — removed during review as misleading (only one document per session), then restored when each chunk was labeled with its source filename, making citation accurate and meaningful. (`doc/CHANGES.md` — "Prompt extracted to file")

### Scaling decisions

Scaling for multiple users means authentication. Authentication means per-user storage. Per-user storage means persistence. Persistence means a caching layer like Redis. None of these features are useful in isolation — they only make sense as a complete system. Since we weren't building that system, we didn't build any of it.

RAG is the exception — it solves a concrete problem without requiring anything else to be in place. It stands alone, which is why it was the one scaling feature worth adding.

### Development process

Development was AI-assisted throughout. Architecture, scope decisions, and all trade-offs were made by the developer. The LLMs — coder, tester, reviewer — were directed and evaluated, not followed.

**1. Architecture and specification**

Started from the assignment requirements. The architecture was designed first — pipelines, endpoints, component responsibilities, session model. This became the opening section of the specification. The spec was a living document, updated continuously as coding decisions were made, and finalized after the code was complete — at that point it was accurate enough to hand to an independent tester.

**2. Testing**

Testing was not a requirement but was treated as one. The tester received the assignment, specification, test contract (`doc/TEST_CONTRACT.md`), evaluation criteria, and a testing methodology document. It had no access to the source code — by design.

The tester needed to be independent because code context is a lens. A tester that has read the source code inherits the developer's assumptions — it tests what the code does rather than what it was supposed to do. Separation guarantees a distinct context: everything the tester knows about the system comes from the spec and the contract, the same interface any external user would have. That's the only way to prevent the test from repeating the same assumptions the developer already made. This is grounded in personal research and a known failure mode in LLM-assisted testing.

The methodology had four steps: map every feature and boundary; fill edge cases (at the limit, just below, just above); add cascade tests — sequences of operations that might pass individually but fail together; and mutation tests — single-property changes to valid requests. This produced an initial suite of ~50 tests.

Several rounds of back and forth followed. Bugs were found — some at the C level inside PyMuPDF and EasyOCR, fixed by adding safeguards in the Python layer above. Rate limit tests revealed that rate limit values needed to be environment variables so they could be overridden during testing. The final suite was 83 tests.

Once isolated from the code, the tester has to derive everything from the contract. That's both the strength and the risk — if the contract is wrong, the tester's assumptions are wrong. The multi-file example illustrates this: the original contract described `/upload` as accepting files (plural). The tester wrote multi-file tests accordingly. When multi-file was reverted in the implementation, those tests broke. The tester had no way to know — it only knew what the contract said. The contract was amended and re-sent. The mismatch was visible precisely because the tester was isolated: it couldn't silently absorb the implementation change the way a code-aware tester would.

**3. Code review**

A separate reviewer with full code access — assignment, spec, code, and tests via `doc/handover.txt` — identified ~30 issues across four categories: security fixes (path traversal, file size limit), robustness fixes (thread safety, error handling, log path), config improvements (hardcoded values moved to environment variables), and design observations. Each was evaluated and either fixed, deferred, or consciously skipped with documented reasoning. All decisions are in `doc/CHANGES.md`.

**4. Cleanup and Docker**

Working across multiple agents and document versions naturally produces drift — spec wording diverges from code, test descriptions get stale, references to removed features linger. After the LLM rounds were done, a cleanup pass went through all doc and test files to bring everything back into alignment. Then Docker — containerizing the two services so the whole thing runs with one command.

---

## Configuration

All values are environment variables. See `.env.example` for the full list.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required |
| `OPENROUTER_MODEL` | — | Required (e.g. `mistralai/ministral-3b-2410`) |
| `MAX_FILE_SIZE_MB` | 50 | Upload size limit |
| `RETRIEVAL_THRESHOLD` | 0.0 | Min cosine score for retrieved chunks |
| `TOP_K` | 5 | Chunks retrieved per question |
| `RATE_LIMIT_UPLOAD` | 10/minute | Per-IP upload rate limit |
| `RATE_LIMIT_ASK` | 30/minute | Per-IP ask rate limit |

---

## Running tests

See `tests/TESTING.md` for full instructions. Requires a live server at `localhost:8000`.

```bash
# Main suite (high rate limits)
$env:RATE_LIMIT_UPLOAD = "1000/minute"; $env:RATE_LIMIT_ASK = "1000/minute"
.venv\Scripts\uvicorn.exe api.app:app --host 0.0.0.0 --port 8000

.venv\Scripts\python.exe -m pytest tests\ --ignore=tests\test_z_rate_limit.py -v
```

---

## Project structure

```
api/            FastAPI app (upload, ask endpoints)
ingestion/      Text extraction (PyMuPDF + EasyOCR)
pipeline/       Chunker, embedder, retriever
session/        In-memory session store
llm/            OpenRouter backend
prompts/        LLM prompt template
app.py          Streamlit frontend
doc/            Specification, changes log, test contract
tests/          Black-box test suite
test_docs/      Sample documents for testing
```
