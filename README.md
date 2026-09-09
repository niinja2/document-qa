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
- An [OpenRouter](https://openrouter.ai) API key and a model name (e.g. `mistralai/mistral-small-24b`)

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
OPENROUTER_MODEL=mistralai/mistral-small-24b
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

## Approach

### Architecture

Two-process system: FastAPI backend + Streamlit frontend. Both containerised via Docker Compose.

RAG pipeline:
- **Chunker** — 500-word chunks, 50-word overlap
- **Embedder** — `BAAI/bge-base-en-v1.5` (768-dim, L2-normalized), served via `sentence-transformers`
- **Index** — FAISS `IndexFlatIP` (cosine similarity on normalized vectors)
- **Retriever** — top-5 chunks by cosine score, filtered by `RETRIEVAL_THRESHOLD`
- **LLM** — OpenRouter (model configurable via `.env`)

Sessions are stored in RAM, keyed by UUID. No persistence across restarts.

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

### Development process

Development was AI-assisted. Architecture and key decisions were made by the developer; code was generated iteratively through a coding assistant. After the initial implementation, a separate critical review pass identified ~30 issues — each was evaluated and either fixed, deferred, or consciously skipped with documented reasoning (see `doc/CHANGES.md`).

Testing used an independent tester LLM that had access only to the spec and test contract — not the source code — so tests were derived from the contract, not from implementation assumptions. The test suite covers file types, edge cases, session isolation, cascade state, mutation tests, and rate limiting.

---

## Configuration

All values are environment variables. See `.env.example` for the full list.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required |
| `OPENROUTER_MODEL` | — | Required (e.g. `mistralai/mistral-small-24b`) |
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
