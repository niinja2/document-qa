# Document QA — Implementation Specification

## Architecture

Two-process system: FastAPI backend + Streamlit frontend, communicating over HTTP.

---

## Flow Diagrams

### POST /upload

```
Streamlit
    │ file + UUID
FastAPI
    │
extractor.py (PyMuPDF / EasyOCR)
    │ text
chunker.py → 500-word chunks, 50-word overlap
    │ chunks
embedder.py → BGE-base-en-v1.5 embeddings (768-dim, normalized)
    │ embeddings
FAISS IndexFlatIP → index
    │
store.py → save(UUID, chunks, index)
    │
FastAPI → {"status": "ok"}
    │
Streamlit → "upload done"
```

### POST /ask

```
Streamlit
    │ question + UUID
FastAPI
    │
store.py → load(UUID) → chunks + index
    │
embedder.py → embed(question)
    │
retriever.py → FAISS search → top-5 chunks
    │ context (chunks labeled with [filename])
llm/backend.py
    │ context + question
OpenRouter API (or DistilBERT fallback)
    │ answer
FastAPI → {"answer": "..."}
    │
Streamlit → displays answer
```

### extractor.py routing

```
extractor.py(file_path)
    │
    ├── .pdf or magic bytes %PDF → extract_pdf(file_path)
    │               │
    │               for each page:
    │                   get_text() → page text
    │                   for each image in page:
    │                       extract_image bytes
    │                       _ocr_from_bytes() → image text
    │               return all text joined
    │
    └── .png/.jpg/etc → extract_image(file_path)
                            EasyOCR → text
                            return text
```

---

## Components

### Backend — api/app.py

- `POST /upload` — accepts one file + session UUID (multipart form). Saves to temp, extracts text, removes temp file. Chunks text, embeds chunks, builds FAISS index, stores in RAM.
- `POST /ask` — accepts session UUID + question (form). Embeds question, retrieves top-5 chunks via FAISS, sends labeled context to LLM, returns answer.
- Startup warmup — embedding model pre-loaded on server start via `lifespan`.
- Rate limiting via slowapi — `RATE_LIMIT_UPLOAD` and `RATE_LIMIT_ASK` (configured in `.env`). Returns HTTP 429 if exceeded.
- JSON structured logging to terminal and `app.log`.

### Text Extraction — ingestion/extractor.py

- PDF: PyMuPDF for text layer, EasyOCR for embedded images
- Images: EasyOCR
- Magic bytes check (`%PDF`) — routes by actual content, not file extension
- Lazy-loaded OCR reader (loaded on first use)

### RAG Pipeline — pipeline/

- `chunker.py` — splits text into 500-word chunks with 50-word overlap
- `embedder.py` — BGE-base-en-v1.5 (768-dim), lazy-loaded, L2-normalized embeddings
- `retriever.py` — FAISS IndexFlatIP cosine search, returns top-K chunks as `{"text": ..., "doc_id": filename}`

### Session Storage — session/store.py

- RAM dict keyed by session UUID
- Stores chunks list + FAISS index per session
- No persistence — cleared on server restart

### LLM — llm/backend.py

- Primary: OpenRouter (model configured via `OPENROUTER_MODEL` in `.env`)
- Fallback: DistilBERT local QA model (when `OPENROUTER_API_KEY` not set)
- Prompt instructs LLM to answer only from provided excerpts and cite the source document

### Frontend — app.py

- Streamlit
- Single-file uploader with Upload button
- Spinner on upload and on ask
- Session UUID generated client-side, stored in `st.session_state`, sent with every request
- `st.form` for question input — supports Enter key submission

### Config — config.py

All values read from environment variables (`.env`), with defaults:

| Variable | Default |
|---|---|
| `CHUNK_SIZE` | 500 |
| `CHUNK_OVERLAP` | 50 |
| `EMBEDDING_MODEL` | BAAI/bge-base-en-v1.5 |
| `TOP_K` | 5 |
| `API_URL` | http://localhost:8000 |
| `RATE_LIMIT_UPLOAD` | 10/minute |
| `RATE_LIMIT_ASK` | 30/minute |

---

## Edge Cases Handled

- No session UUID yet → 404 "upload first"
- Unsupported file type → 422
- Empty or corrupt file (< 100 bytes) → 422, size guard fires before C-level libraries
- Empty text after extraction → 422
- PDF bytes sent with wrong extension → magic bytes check routes correctly
- Rate limit exceeded → HTTP 429

---

## Not Implemented

- Docker / docker-compose
- Persistence (sessions lost on server restart)
- Authentication
- NER
- Redis caching
