import logging
from pipeline.embedder import embed
from config import TOP_K

logger = logging.getLogger(__name__)


def retrieve(
    question: str,
    index,
    chunks: list[dict],
) -> list[dict]:
    question_embedding = embed([question])
    distances, indices = index.search(question_embedding, min(TOP_K, len(chunks)))
    # eg indices = [[3, 7, 1, 0, 5]] , indices[0] = [3, 7, 1, 0, 5]
    hit_positions = []
    for i in indices[0]:
        if i >= 0:  # FAISS returns -1 for empty slots when index has fewer items than TOP_K
            hit_positions.append(i)
    top_chunks = []
    for i in hit_positions:
        top_chunks.append(chunks[i])
    logger.info("Retrieved %d chunks for question: %s", len(top_chunks), question)
    return top_chunks
