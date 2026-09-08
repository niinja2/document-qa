import os
import logging
import requests
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

logger = logging.getLogger(__name__)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

_qa_tokenizer = None
_qa_model = None


def _get_qa_model():
    global _qa_tokenizer, _qa_model
    if _qa_model is None:
        _qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        _qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    return _qa_tokenizer, _qa_model


def _answer_openrouter(text: str, question: str, api_key: str) -> str:
    prompt = (
        f"The following are excerpts from one or more documents. "
        f"Each excerpt is labeled with its source file in brackets.\n\n"
        f"{text}\n\n"
        f"Answer the question using only the information in the excerpts above. "
        f"Be concise. At the end of your answer, state which document(s) you used.\n"
        f"If the answer cannot be found in the excerpts, say so explicitly.\n\n"
        f"Question: {question}"
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    data = response.json()
    if "choices" not in data:
        logger.error("OpenRouter error: %s", data)
        raise RuntimeError(f"OpenRouter error: {data}")
    answer = data["choices"][0]["message"]["content"]
    return answer


def _answer_distilbert(text: str, question: str) -> str:
    tokenizer, model = _get_qa_model()
    inputs = tokenizer(question, text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start = outputs.start_logits.argmax()
    end = outputs.end_logits.argmax() + 1
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    answer = tokenizer.convert_tokens_to_string(tokens)
    return answer


def answer(text: str, question: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        logger.info("Using OpenRouter (%s)", OPENROUTER_MODEL)
        return _answer_openrouter(text, question, api_key)
    else:
        logger.info("Using DistilBERT")
        return _answer_distilbert(text, question)
