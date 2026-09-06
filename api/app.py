import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, Form, HTTPException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)

from ingestion import extractor
from session import store
from llm import backend

app = FastAPI()


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

    store.save(session_id, text)
    return {"status": "ok"}


@app.post("/ask")
async def ask(
    session_id: str = Form(...),
    question: str = Form(...),
):
    text = store.load(session_id)

    if text is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a document first.")

    result = backend.answer(text, question)
    return {"answer": result}
