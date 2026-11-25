from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chunk_size: int = 800
    chunk_overlap: int = 120

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 8
    rerank_top_n: int = 3

    llm_provider: Literal["local", "openai", "groq"] = "local"
    local_model: str = "google/flan-t5-base"
    openai_model: str = "gpt-4o-mini"
    groq_model: str = "llama-3.1-8b-instant"
    openai_api_key: str | None = None
    groq_api_key: str | None = None

    data_dir: str = "./data"
    max_upload_mb: int = 20
    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
