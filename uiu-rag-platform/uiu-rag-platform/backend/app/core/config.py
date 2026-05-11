"""
UIU Academic RAG Platform — Core Configuration
Updated to use Groq (free) instead of OpenAI.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "UIU Academic RAG Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    SECRET_KEY: str = Field(..., min_length=32)
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── PostgreSQL + pgvector ─────────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: str = Field(...)
    POSTGRES_DB: str = "uiu_rag"
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 40
    POSTGRES_POOL_TIMEOUT: int = 30

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600

    # ── Google Drive (optional) ───────────────────────────────────────────────
    GOOGLE_SERVICE_ACCOUNT_JSON: str = "./secrets/google_service_account.json"
    GOOGLE_DRIVE_ROOT_FOLDER_ID: str = "1rUXi7uuqKKZ50ktWzTkTd8uLIKXEWqar"
    DRIVE_SYNC_INTERVAL_MINUTES: int = 60
    DRIVE_MAX_FILE_SIZE_MB: int = 100

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_MAX_RETRIES: int = 3
    EMBEDDING_DEVICE: str = "cpu"

    # ── Reranker ─────────────────────────────────────────────────────────────
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    RERANKER_TOP_K: int = 5

    # ── Groq LLM (FREE) ───────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(...)
    # Available free Groq models:
    # - llama-3.1-8b-instant   (fastest, good quality)
    # - llama-3.1-70b-versatile (slower, best quality)
    # - mixtral-8x7b-32768     (good for long context)
    # - gemma2-9b-it           (Google's model)
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_STREAMING: bool = True
    LLM_TIMEOUT: int = 60
    LLM_MAX_RETRIES: int = 3

    # ── Retrieval ─────────────────────────────────────────────────────────────
    VECTOR_SEARCH_TOP_K: int = 20
    BM25_SEARCH_TOP_K: int = 20
    HYBRID_ALPHA: float = 0.6
    FINAL_TOP_K: int = 5
    MIN_RELEVANCE_SCORE: float = 0.35

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MIN_CHUNK_SIZE: int = 50

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    @field_validator("EMBEDDING_DIMENSION")
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        if v not in {384, 768, 1024, 1536}:
            raise ValueError("Embedding dimension must be one of {384, 768, 1024, 1536}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
