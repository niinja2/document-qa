_sessions = {}


def save(session_id: str, text: str) -> None:
    _sessions[session_id] = text


def load(session_id: str) -> str | None:
    return _sessions.get(session_id)
