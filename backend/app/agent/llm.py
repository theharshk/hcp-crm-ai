"""
Thin LLM wrapper — auto-selects Gemini (preferred, if GEMINI_API_KEY is set) or Groq.

Gemini models:
  - gemini-2.0-flash -> fast, high rate limits, used for conversational agent loop
  - gemini-1.5-pro   -> heavier model, used when stronger reasoning is needed

Groq fallback models:
  - llama-3.1-8b-instant   -> fast, cheap, used for the conversational agent loop
  - llama-3.3-70b-versatile -> heavier model, used for stronger reasoning steps
"""
import google.generativeai as genai
from groq import Groq
from app.config import settings


def chat_completion(messages: list[dict], model: str | None = None, temperature: float = 0.0) -> str:
    """Route LLM completion through Gemini if API key is set, else fall back to Groq."""
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        target_model = model or settings.GEMINI_CHAT_MODEL
        gemini_model = genai.GenerativeModel(
            model_name=target_model,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        # Convert OpenAI-style messages list to a single prompt string
        prompt_parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            if role == "SYSTEM":
                prompt_parts.append(f"[SYSTEM INSTRUCTIONS]\n{content}")
            else:
                prompt_parts.append(f"{role}: {content}")
        prompt = "\n\n".join(prompt_parts)
        response = gemini_model.generate_content(prompt)
        return response.text
    else:
        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model or settings.GROQ_CHAT_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content


def reasoning_completion(messages: list[dict], temperature: float = 0.0) -> str:
    """Escalate to the larger model for harder reasoning steps."""
    if settings.GEMINI_API_KEY:
        # Use gemini-1.5-pro for heavy reasoning when on Gemini
        return chat_completion(messages, model="gemini-1.5-pro", temperature=temperature)
    else:
        return chat_completion(messages, model=settings.GROQ_REASONING_MODEL, temperature=temperature)
