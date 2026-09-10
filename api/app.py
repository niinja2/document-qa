from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import os
import tempfile
import logging
from contextlib import asynccontextmanager
from pythonjsonlogger.json import JsonFormatter
from fastapi import FastAPI, UploadFile, Form, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import requests

from config import RATE_LIMIT_UPLOAD, RATE_LIMIT_ASK, MAX_FILE_SIZE_MB
from ingestion import extractor
from session import store
from llm import backend
from pipeline import chunker, embedder, retriever
import faiss

json_formatter = JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(json_formatter)

_log_path = os.getenv("LOG_FILE", os.path.join(tempfile.gettempdir(), "app.log"))
_file_handler = logging.FileHandler(_log_path)
_file_handler.setFormatter(json_formatter)

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler], force=True)

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder.embed(["warmup"])
    extractor._get_ocr_reader()
    logger.info("Models warmed up")
    yield


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/upload")
@limiter.limit(RATE_LIMIT_UPLOAD)
def upload(
    request: Request,
    file: UploadFile,
    session_id: str = Form(...),
):
    suffix = Path(file.filename).suffix
    contents = file.file.read()

    if len(contents) < 100:
        raise HTTPException(status_code=422, detail="File is too small or empty")

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB",
        )

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(contents)
    tmp.close()
    tmp_path = tmp.name

    try:
        text = extractor.extract(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=422, detail=str(e))

    os.remove(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=422, detail=f"Could not extract text from {file.filename}")

    chunks = chunker.chunk_text(text)
    embeddings = embedder.embed(chunks)

    tagged_chunks = [{"text": chunk, "doc_id": file.filename} for chunk in chunks]

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    store.save(session_id, tagged_chunks, index)
    logger.info("Uploaded session %s: %d chunks", session_id, len(tagged_chunks))
    return {"status": "ok"}


@app.post("/ask")
@limiter.limit(RATE_LIMIT_ASK)
def ask(
    request: Request,
    session_id: str = Form(...),
    question: str = Form(...),
):
    session = store.load(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a document first.")

    chunks = session["chunks"]
    index = session["index"]

    top_chunks = retriever.retrieve(question, index, chunks)
    context_parts = [f"[{c['doc_id']}]\n{c['text']}" for c in top_chunks]
    context = "\n\n".join(context_parts)

    try:
        result = backend.answer(context, question)
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="LLM service timed out. Please try again.")
    except requests.ConnectionError:
        raise HTTPException(status_code=502, detail="LLM service unreachable.")
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM service error (HTTP {e.response.status_code}).")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"answer": result}
