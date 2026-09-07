from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import os
import logging
from contextlib import asynccontextmanager
from pythonjsonlogger.json import JsonFormatter
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException

from ingestion import extractor
from session import store
from llm import backend
from pipeline import chunker, embedder, retriever
import faiss

json_formatter = JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(json_formatter)

_file_handler = logging.FileHandler("app.log")
_file_handler.setFormatter(json_formatter)

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder.embed(["warmup"])
    logger.info("Embedding model warmed up")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/upload")
async def upload(
    files: list[UploadFile],
    session_id: str = Form(...),
):
    all_chunks = []
    all_embeddings = []

    for file in files:
        suffix = Path(file.filename).suffix
        tmp_path = f"/tmp/{session_id}_{file.filename}{suffix}"

        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        text = extractor.extract(tmp_path)
        os.remove(tmp_path)

        if not text.strip():
            raise HTTPException(status_code=422, detail=f"Could not extract text from {file.filename}")

        chunks = chunker.chunk_text(text)
        embeddings = embedder.embed(chunks)

        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    combined = np.vstack(all_embeddings)
    dimension = combined.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(combined)

    store.save(session_id, all_chunks, index)
    logger.info("Uploaded session %s: %d files, %d chunks", session_id, len(files), len(all_chunks))
    return {"status": "ok"}


@app.post("/ask")
async def ask(
    session_id: str = Form(...),
    question: str = Form(...),
):
    session = store.load(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a document first.")

    chunks = session["chunks"]
    index = session["index"]

    top_chunks = retriever.retrieve(question, index, chunks)
    context = "\n\n".join(top_chunks)

    result = backend.answer(context, question)
    return {"answer": result}
