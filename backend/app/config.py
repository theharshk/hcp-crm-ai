"""
Central configuration for the HCP CRM backend.
All secrets are read from environment variables (.env) — never hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Groq / LLM ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_CHAT_MODEL: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
    GROQ_REASONING_MODEL: str = os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")

    # --- Gemini / Google AI ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

    # --- Database ---
    # Works with either Postgres or MySQL — just swap the URL scheme.
    # Postgres: postgresql+psycopg2://user:password@localhost:5432/hcp_crm
    # MySQL:    mysql+pymysql://user:password@localhost:3306/hcp_crm
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/hcp_crm"
    )

    # --- App ---
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")


settings = Settings()

active_provider = "Gemini" if settings.GEMINI_API_KEY else "Groq"
print(f"[INFO] LLM Provider: {active_provider}")

if not settings.GEMINI_API_KEY and not settings.GROQ_API_KEY:
    print(
        "[WARN] Neither GEMINI_API_KEY nor GROQ_API_KEY is set. "
        "Add one to backend/.env — the agent will fail on any LLM call until this is set."
    )
