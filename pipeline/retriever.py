import logging
from pipeline.embedder import embed
from config import TOP_K, RETRIEVAL_THRESHOLD

logger = logging.getLogger(__name__)


def retrieve(
    question: str,
    index,
    chunks: list[dict],
) -> list[dict]:
    question_embedding = embed([question])
    distances, indices = index.search(question_embedding, min(TOP_K, len(chunks)))
    # indices = [[3, 7, 1, 0, 5]], distances = [[0.91, 0.74, ...]]
    hit_positions = []
    for score, i in zip(distances[0], indices[0]):
        if i >= 0 and score >= RETRIEVAL_THRESHOLD:
            hit_positions.append(i)
    top_chunks = []
    for i in hit_positions:
        top_chunks.append(chunks[i])
    logger.info("Retrieved %d chunks for question: %s", len(top_chunks), question)
    return top_chunks
