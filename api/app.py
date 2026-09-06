from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import os
import logging
from fastapi import FastAPI, UploadFile, Form, HTTPException

from ingestion import extractor
from session import store
from llm import backend
from pipeline import chunker, embedder, retriever
import faiss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)

logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event("startup")
async def warm_up():
    embedder.embed(["warmup"])
    logger.info("Embedding model warmed up")


@app.post("/upload")
async def upload(
    file: UploadFile,
    session_id: str = Form(...),
):
    suffix = Path(file.filename).suffix
    tmp_path = f"/tmp/{session_id}{suffix}"

    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    text = extractor.extract(tmp_path)
    os.remove(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file")

    chunks = chunker.chunk_text(text)
    embeddings = embedder.embed(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    store.save(session_id, chunks, index)
    logger.info("Uploaded session %s: %d chunks", session_id, len(chunks))
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
