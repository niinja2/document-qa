_sessions = {}


def save(
    session_id: str,
    chunks: list[str],
    index,
) -> None:
    _sessions[session_id] = {
        "chunks": chunks,
        "index": index,
    }


def load(session_id: str) -> dict | None:
    return _sessions.get(session_id)
