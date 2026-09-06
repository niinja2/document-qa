from pipeline.embedder import embed
from config import TOP_K


def retrieve(
    question: str,
    index,
    chunks: list[str],
) -> list[str]:
    question_embedding = embed([question])
    distances, indices = index.search(question_embedding, TOP_K)
    top_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return top_chunks
