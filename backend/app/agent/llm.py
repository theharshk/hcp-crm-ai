"""
Thin wrapper around the Groq API.

- gemma2-9b-it        -> fast, cheap, used for the conversational agent loop
- llama-3.3-70b-versatile -> heavier model, used when we need stronger
                             reasoning (e.g. resolving ambiguous edits,
                             complex multi-entity extraction)
"""
from groq import Groq
from app.config import settings

_client = Groq(api_key=settings.GROQ_API_KEY)


def chat_completion(messages: list[dict], model: str | None = None, temperature: float = 0.2) -> str:
    model = model or settings.GROQ_CHAT_MODEL
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def reasoning_completion(messages: list[dict], temperature: float = 0.1) -> str:
    """Escalate to the larger model for harder reasoning steps."""
    return chat_completion(messages, model=settings.GROQ_REASONING_MODEL, temperature=temperature)
