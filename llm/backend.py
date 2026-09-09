import os
import logging
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

_prompt_path = Path(__file__).parent.parent / "prompts" / "qa_prompt.txt"
if not _prompt_path.exists():
    raise FileNotFoundError(f"Prompt template not found: {_prompt_path}")
_PROMPT_TEMPLATE = _prompt_path.read_text(encoding="utf-8")


def answer(text: str, question: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Set it in .env to use the QA service.")
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is not set. Set it in .env to use the QA service.")
    logger.info("Using OpenRouter (%s)", model)
    prompt = _PROMPT_TEMPLATE.format(context=text, question=question)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=(5, 60),
    )
    response.raise_for_status()
    data = response.json()
    if "choices" not in data:
        logger.error("OpenRouter error: %s", data)
        raise RuntimeError(f"OpenRouter error: {data}")
    result = data["choices"][0]["message"]["content"]
    return result
