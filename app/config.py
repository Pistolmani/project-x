"""Typed settings read from the environment / `.env` (decision A5). Secrets never have defaults."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # Free tier, measured 2026-10-24: gemini-3.5-flash allows 20 requests/day (5/min), far too few for the
    # eval, so flash-lite investigates and flash is a second Gemini quota before Groq (1000/day, 8k tok/min).
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_model: str = "gemini-3.5-flash"
    groq_model: str = "openai/gpt-oss-120b"
    llm_timeout_s: float = 30
    llm_retries: int = 2  # per provider, for retryable errors only
    llm_max_wait_s: float = 60  # longest provider-requested wait before moving to the next provider

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"

    chunk_size: int = 256  # tokens, counted with the dense model's own tokenizer (A2)
    chunk_overlap: int = 32  # fixed strategy only

    chunks_collection: str = "chunks"
    documents_collection: str = "documents"

    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"
    cache_path: Path = PROJECT_ROOT / "data" / "cache.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
