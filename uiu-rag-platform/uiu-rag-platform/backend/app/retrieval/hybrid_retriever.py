"""
UIU Academic RAG Platform — Hybrid Retrieval System

Pipeline:
  Query
    ├── Vector Search (pgvector cosine similarity)
    └── BM25 Full-Text Search (PostgreSQL tsvector)
         ↓
  Reciprocal Rank Fusion (RRF)
         ↓
  BGE Reranker (cross-encoder relevance scoring)
         ↓
  Top-K Chunks with scores

Design decisions:
  - RRF is rank-based (not score-based), so vector scores and BM25 scores
    don't need normalization before merging.
  - Metadata filters (course, document_type) applied BEFORE vector search
    to reduce candidate space and improve retrieval precision.
  - Reranker uses full cross-encoder (not bi-encoder) for superior relevance.
  - Async throughout — no blocking DB or model calls in request path.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.embeddings.pipeline import embedding_pipeline
from app.models.document import DocumentChunk

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its score and metadata for prompt construction."""
    chunk_id: str
    content: str
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    # Metadata
    course: str = ""
    document_type: str = ""
    source_filename: str = ""
    page_number: int | None = None
    slide_number: int | None = None
    section_heading: str | None = None
    topic: str | None = None
    semester: str | None = None

    @property
    def citation_ref(self) -> str:
        parts = [self.source_filename]
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        if self.slide_number:
            parts.append(f"slide {self.slide_number}")
        if self.section_heading:
            parts.append(f"§ {self.section_heading}")
        return " — ".join(parts)


@dataclass
class QueryFilter:
    """Filters applied to retrieval — enables scoped course/type queries."""
    course: str | None = None
    document_types: list[str] | None = None   # e.g. ["lecture_slide", "notes"]
    semester: str | None = None
    topic: str | None = None


# ── Vector Search ─────────────────────────────────────────────────────────────
async def vector_search(
    session: AsyncSession,
    query_embedding: list[float],
    filters: QueryFilter,
    top_k: int = settings.VECTOR_SEARCH_TOP_K,
) -> list[tuple[str, float]]:
    """
    pgvector cosine similarity search with optional metadata filters.
    Returns list of (chunk_id, similarity_score).
    """
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    # Build WHERE clause dynamically based on filters
    conditions = ["1=1"]
    params: dict[str, Any] = {
        "embedding": embedding_str,
        "top_k": top_k,
        "min_score": settings.MIN_RELEVANCE_SCORE,
    }

    if filters.course:
        conditions.append("course = :course")
        params["course"] = filters.course

    if filters.document_types:
        conditions.append("document_type = ANY(:doc_types)")
        params["doc_types"] = filters.document_types

    if filters.semester:
        conditions.append("semester = :semester")
        params["semester"] = filters.semester

    where_clause = " AND ".join(conditions)

    sql = text(f"""
        SELECT
            id::text,
            1 - (embedding <=> :embedding::vector) AS similarity
        FROM document_chunks
        WHERE {where_clause}
            AND 1 - (embedding <=> :embedding::vector) >= :min_score
        ORDER BY similarity DESC
        LIMIT :top_k
    """)

    try:
        result = await session.execute(sql, params)
        rows = result.fetchall()
        return [(row[0], float(row[1])) for row in rows]
    except Exception as e:
        logger.error("Vector search failed", error=str(e))
        raise RetrievalError(f"Vector search failed: {e}")


# ── BM25 Full-Text Search ─────────────────────────────────────────────────────
def _clean_query_for_tsquery(query: str) -> str:
    """
    Sanitize query for PostgreSQL plainto_tsquery.
    Remove special chars that would cause parse errors.
    """
    clean = re.sub(r"[^\w\s]", " ", query)
    return " ".join(clean.split())


async def bm25_search(
    session: AsyncSession,
    query: str,
    filters: QueryFilter,
    top_k: int = settings.BM25_SEARCH_TOP_K,
) -> list[tuple[str, float]]:
    """
    PostgreSQL full-text search using tsvector and ts_rank_cd.
    Returns list of (chunk_id, rank_score).
    """
    clean_query = _clean_query_for_tsquery(query)
    if not clean_query.strip():
        return []

    conditions = ["1=1"]
    params: dict[str, Any] = {
        "query": clean_query,
        "top_k": top_k,
    }

    if filters.course:
        conditions.append("course = :course")
        params["course"] = filters.course

    if filters.document_types:
        conditions.append("document_type = ANY(:doc_types)")
        params["doc_types"] = filters.document_types

    where_clause = " AND ".join(conditions)

    sql = text(f"""
        SELECT
            id::text,
            ts_rank_cd(
                to_tsvector('english', content),
                plainto_tsquery('english', :query),
                32  -- normalization: rank / (rank + 1)
            ) AS rank
        FROM document_chunks
        WHERE {where_clause}
            AND to_tsvector('english', content) @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :top_k
    """)

    try:
        result = await session.execute(sql, params)
        rows = result.fetchall()
        return [(row[0], float(row[1])) for row in rows]
    except Exception as e:
        logger.error("BM25 search failed", error=str(e))
        return []  # BM25 failure is non-fatal; fall back to vector-only


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
    alpha: float = settings.HYBRID_ALPHA,
) -> list[tuple[str, float]]:
    """
    Merge vector and BM25 results using Reciprocal Rank Fusion.

    RRF score = alpha * 1/(k + rank_vector) + (1-alpha) * 1/(k + rank_bm25)

    k=60 is the standard RRF constant that prevents very high scores for top ranks.
    alpha controls the vector vs BM25 weight.
    """
    scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(vector_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0) + alpha * (1 / (k + rank))

    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0) + (1 - alpha) * (1 / (k + rank))

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged


# ── BGE Reranker ──────────────────────────────────────────────────────────────
class BGEReranker:
    """
    Cross-encoder reranker using bge-reranker-base.
    Superior to bi-encoder for relevance judgment because
    it jointly encodes query + passage.
    Model loaded once and cached in memory.
    """

    _model: Any = None
    _lock = asyncio.Lock()

    @classmethod
    async def load(cls) -> None:
        async with cls._lock:
            if cls._model is None:
                try:
                    from sentence_transformers import CrossEncoder

                    cls._model = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: CrossEncoder(
                            settings.RERANKER_MODEL_NAME,
                            device=settings.EMBEDDING_DEVICE,
                        ),
                    )
                    logger.info("Reranker model loaded", model=settings.RERANKER_MODEL_NAME)
                except Exception as e:
                    logger.error("Reranker load failed", error=str(e))
                    cls._model = None

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Score each (query, passage) pair and sort by relevance."""
        if cls._model is None:
            await BGEReranker.load()

        if self._model is None or not chunks:
            return chunks

        pairs = [(query, chunk.content) for chunk in chunks]

        try:
            scores = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._model.predict(pairs, show_progress_bar=False),
            )

            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)

            reranked = sorted(chunks, key=lambda c: c.rerank_score or 0, reverse=True)
            return reranked[: settings.RERANKER_TOP_K]

        except Exception as e:
            logger.warning("Reranking failed, using RRF order", error=str(e))
            return chunks[: settings.RERANKER_TOP_K]


reranker = BGEReranker()


# ── Hybrid Retriever ──────────────────────────────────────────────────────────
class HybridRetriever:
    """
    Orchestrates the full retrieval pipeline.
    Single entry point for the RAG query flow.
    """

    async def retrieve(
        self,
        session: AsyncSession,
        query: str,
        filters: QueryFilter | None = None,
        top_k: int = settings.FINAL_TOP_K,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval: vector + BM25 → RRF → rerank → top-k.
        """
        if filters is None:
            filters = QueryFilter()

        # 1. Embed query
        query_embedding = await embedding_pipeline.embed_query(query)

        # 2. Parallel vector + BM25 search
        vector_results, bm25_results = await asyncio.gather(
            vector_search(session, query_embedding, filters),
            bm25_search(session, query, filters),
        )

        logger.info(
            "Retrieval candidates",
            vector_results=len(vector_results),
            bm25_results=len(bm25_results),
        )

        if not vector_results and not bm25_results:
            return []

        # 3. RRF fusion
        fused = reciprocal_rank_fusion(vector_results, bm25_results)

        # 4. Fetch chunk content for top candidates (before reranking)
        candidate_ids = [chunk_id for chunk_id, _ in fused[:max(top_k * 4, 20)]]
        chunks = await self._fetch_chunks(session, candidate_ids, dict(fused))

        # 5. Rerank
        reranked = await reranker.rerank(query, chunks)

        logger.info(
            "Retrieval complete",
            returned_chunks=len(reranked),
            query_preview=query[:80],
        )
        return reranked

    async def _fetch_chunks(
        self,
        session: AsyncSession,
        chunk_ids: list[str],
        rrf_scores: dict[str, float],
    ) -> list[RetrievedChunk]:
        """Fetch chunk rows from DB and hydrate RetrievedChunk objects."""
        if not chunk_ids:
            return []

        sql = text("""
            SELECT
                id::text, content, course, document_type,
                source_filename, page_number, slide_number,
                section_heading, topic, semester
            FROM document_chunks
            WHERE id::text = ANY(:ids)
        """)

        result = await session.execute(sql, {"ids": chunk_ids})
        rows = result.fetchall()

        chunks = []
        for row in rows:
            chunk_id = row[0]
            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=row[1],
                score=rrf_scores.get(chunk_id, 0),
                rrf_score=rrf_scores.get(chunk_id, 0),
                course=row[2] or "",
                document_type=row[3] or "",
                source_filename=row[4] or "",
                page_number=row[5],
                slide_number=row[6],
                section_heading=row[7],
                topic=row[8],
                semester=row[9],
            ))

        return chunks


# Singleton
hybrid_retriever = HybridRetriever()
