"""
UIU Academic RAG Platform — Database Session & pgvector Setup
Async SQLAlchemy engine with connection pooling.
pgvector extension auto-registered for cosine similarity search.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Async Engine ──────────────────────────────────────────────────────────────
# QueuePool is the default; we tune pool_size for production load.
# echo=False in production — SQL logging is expensive.
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.POSTGRES_POOL_SIZE,
    max_overflow=settings.POSTGRES_MAX_OVERFLOW,
    pool_timeout=settings.POSTGRES_POOL_TIMEOUT,
    pool_pre_ping=True,       # detect stale connections before use
    pool_recycle=3600,         # recycle connections every hour
    echo=settings.DEBUG,
    future=True,
)

# ── Session Factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,    # prevent lazy-load after commit
    autocommit=False,
    autoflush=False,
)


# ── Base Model ────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ── Dependency ────────────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: provides a scoped async DB session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Database Initialization ───────────────────────────────────────────────────
async def init_db() -> None:
    """
    Initialize the database:
    1. Enable pgvector extension
    2. Create all tables
    3. Create vector index if not exists
    """
    async with engine.begin() as conn:
        # Enable pgvector
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))  # for BM25-like text search

        # Create all ORM-defined tables
        await conn.run_sync(Base.metadata.create_all)

        # Create HNSW index for fast approximate nearest-neighbor search
        # HNSW is preferred over IVFFlat for latency-sensitive retrieval
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))

        # GIN index for full-text search (BM25 hybrid)
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_content_gin
            ON document_chunks
            USING gin(to_tsvector('english', content))
        """))

        # Composite index for metadata filtering
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_course_type
            ON document_chunks (course, document_type)
        """))

    logger.info("Database initialized with pgvector, GIN, and HNSW indexes")


async def health_check() -> bool:
    """Verify database connectivity — used by health endpoint."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False
