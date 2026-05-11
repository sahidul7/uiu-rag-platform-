"""
UIU Academic RAG Platform — Embedding Pipeline
BAAI/bge-small-en-v1.5 with:
  - Batch processing (GPU-friendly)
  - Redis-based embedding cache (avoid re-embedding on reingestion)
  - Async-safe execution (thread pool for CPU-bound inference)
  - Exponential backoff retry
  - Instruction prefix for BGE models

BGE model requires an instruction prefix for retrieval tasks:
  Passage: "Represent this sentence for searching relevant passages: {text}"
  Query:   "Represent this question for searching relevant passages: {text}"
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.exceptions import EmbeddingError, EmbeddingModelNotLoadedError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── BGE instruction prefixes ──────────────────────────────────────────────────
BGE_PASSAGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "


class EmbeddingModel:
    """
    Thread-safe singleton wrapper around SentenceTransformer.
    Lazy-loaded at first use (avoids startup delay).
    Model lives in memory for the lifetime of the process.
    """

    _model: SentenceTransformer | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def load(cls) -> SentenceTransformer:
        async with cls._lock:
            if cls._model is None:
                logger.info(
                    "Loading embedding model",
                    model=settings.EMBEDDING_MODEL_NAME,
                    device=settings.EMBEDDING_DEVICE,
                )
                try:
                    # Run blocking model load in thread pool
                    cls._model = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: SentenceTransformer(
                            settings.EMBEDDING_MODEL_NAME,
                            device=settings.EMBEDDING_DEVICE,
                        ),
                    )
                    logger.info("Embedding model loaded successfully")
                except Exception as e:
                    raise EmbeddingModelNotLoadedError(
                        f"Failed to load {settings.EMBEDDING_MODEL_NAME}: {e}"
                    )
        return cls._model  # type: ignore[return-value]

    @classmethod
    async def get(cls) -> SentenceTransformer:
        if cls._model is None:
            await cls.load()
        return cls._model  # type: ignore[return-value]


# ── Redis embedding cache ─────────────────────────────────────────────────────
class EmbeddingCache:
    """
    Content-hash → embedding vector cache.
    Avoids re-embedding unchanged chunks during incremental sync.
    Falls back gracefully if Redis unavailable.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._prefix = "emb:v1:"
        self._ttl = 30 * 24 * 3600  # 30 days

    async def _get_client(self) -> Any | None:
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(
                    settings.REDIS_URL,
                    decode_responses=False,
                    socket_timeout=2,
                )
                await self._client.ping()
            except Exception as e:
                logger.warning("Redis unavailable, embedding cache disabled", error=str(e))
                self._client = None
        return self._client

    def _cache_key(self, content_hash: str) -> str:
        return f"{self._prefix}{content_hash}"

    async def get(self, content_hash: str) -> list[float] | None:
        client = await self._get_client()
        if not client:
            return None
        try:
            data = await client.get(self._cache_key(content_hash))
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    async def set(self, content_hash: str, embedding: list[float]) -> None:
        client = await self._get_client()
        if not client:
            return
        try:
            await client.setex(
                self._cache_key(content_hash),
                self._ttl,
                json.dumps(embedding),
            )
        except Exception as e:
            logger.warning("Cache write failed", error=str(e))

    async def mget(self, hashes: list[str]) -> dict[str, list[float]]:
        """Batch get from cache."""
        client = await self._get_client()
        if not client:
            return {}
        try:
            keys = [self._cache_key(h) for h in hashes]
            values = await client.mget(*keys)
            return {
                h: json.loads(v)
                for h, v in zip(hashes, values)
                if v is not None
            }
        except Exception:
            return {}


embedding_cache = EmbeddingCache()


# ── Embedding Pipeline ────────────────────────────────────────────────────────
class EmbeddingPipeline:
    """
    Production embedding pipeline:
    1. Check cache for each content hash
    2. Batch encode cache misses
    3. Write new embeddings to cache
    4. Return ordered list aligned with input
    """

    def __init__(self) -> None:
        self._batch_size = settings.EMBEDDING_BATCH_SIZE

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(settings.EMBEDDING_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _encode_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Encode a batch of texts. Runs inference in thread pool."""
        model = await EmbeddingModel.get()
        instruction = BGE_QUERY_INSTRUCTION if is_query else BGE_PASSAGE_INSTRUCTION
        prefixed = [instruction + t for t in texts]

        try:
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.encode(
                    prefixed,
                    normalize_embeddings=True,   # cosine similarity = dot product
                    show_progress_bar=False,
                    batch_size=min(self._batch_size, len(prefixed)),
                    convert_to_numpy=True,
                ),
            )
            return embeddings.tolist()  # type: ignore[union-attr]
        except Exception as e:
            raise EmbeddingError(f"Embedding inference failed: {e}")

    async def embed_chunks(
        self,
        contents: list[str],
        content_hashes: list[str],
    ) -> list[list[float]]:
        """
        Embed a list of chunk contents.
        Cache-aware: only encodes cache misses.
        Returns embeddings in same order as input.
        """
        assert len(contents) == len(content_hashes)

        # 1. Cache lookup
        cached = await embedding_cache.mget(content_hashes)

        # 2. Identify cache misses
        misses: list[tuple[int, str, str]] = []  # (original_idx, content, hash)
        for i, (content, hash_) in enumerate(zip(contents, content_hashes)):
            if hash_ not in cached:
                misses.append((i, content, hash_))

        logger.info(
            "Embedding batch",
            total=len(contents),
            cache_hits=len(cached),
            cache_misses=len(misses),
        )

        # 3. Encode misses in batches
        if misses:
            miss_texts = [m[1] for m in misses]
            miss_hashes = [m[2] for m in misses]

            all_embeddings: list[list[float]] = []
            for batch_start in range(0, len(miss_texts), self._batch_size):
                batch = miss_texts[batch_start : batch_start + self._batch_size]
                batch_embeddings = await self._encode_batch(batch, is_query=False)
                all_embeddings.extend(batch_embeddings)

            # 4. Write to cache and merge results
            cache_writes = []
            for (orig_idx, content, hash_), emb in zip(misses, all_embeddings):
                cached[hash_] = emb
                cache_writes.append(embedding_cache.set(hash_, emb))

            await asyncio.gather(*cache_writes, return_exceptions=True)

        # 5. Assemble in original order
        return [cached[h] for h in content_hashes]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a user query with BGE query instruction prefix."""
        embeddings = await self._encode_batch([query], is_query=True)
        return embeddings[0]

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single passage (utility method)."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        results = await self.embed_chunks([text], [content_hash])
        return results[0]


# Singleton
embedding_pipeline = EmbeddingPipeline()
