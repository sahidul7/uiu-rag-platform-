"""
UIU Academic RAG Platform — Core RAG Service
Orchestrates the complete query pipeline from user question to cited response.

Pipeline:
  User Question + Course Filter
    ↓ QueryProcessor (expand, clean)
    ↓ HybridRetriever (vector + BM25 + RRF + rerank)
    ↓ PromptBuilder (context injection + citation formatting)
    ↓ LLMInference (streaming or blocking)
    ↓ CitationExtractor
    ↓ CitedResponse
"""
from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import NoRelevantContextError
from app.core.logging import get_logger
from app.llm.inference import InferenceResult, llm_inference
from app.prompts.builder import citation_extractor, prompt_builder
from app.retrieval.hybrid_retriever import QueryFilter, RetrievedChunk, hybrid_retriever

logger = get_logger(__name__)


@dataclass
class RAGQueryRequest:
    question: str
    course: str | None = None
    document_types: list[str] | None = None
    semester: str | None = None
    conversation_id: str | None = None
    conversation_history: list[dict[str, str]] | None = None
    stream: bool = False


@dataclass
class RAGQueryResponse:
    answer: str
    citations: list[dict[str, Any]]
    source_map: dict[int, str]
    retrieved_chunks: list[dict[str, Any]]
    latency_ms: int
    model: str
    was_grounded: bool       # True if answer came from retrieved context
    was_compressed: bool     # True if context was truncated to fit token budget
    context_count: int


class RAGService:
    """
    The central service binding all RAG components.
    This is the single point of entry for query processing.
    """

    async def query(
        self,
        db: AsyncSession,
        request: RAGQueryRequest,
    ) -> RAGQueryResponse:
        """Full blocking RAG query — returns complete cited response."""
        start_time = time.monotonic()

        # 1. Build retrieval filter
        query_filter = QueryFilter(
            course=request.course,
            document_types=request.document_types,
            semester=request.semester,
        )

        # 2. Hybrid retrieval
        retrieved_chunks = await hybrid_retriever.retrieve(
            session=db,
            query=request.question,
            filters=query_filter,
            top_k=settings.FINAL_TOP_K,
        )

        logger.info(
            "RAG retrieval done",
            question_preview=request.question[:60],
            chunks_retrieved=len(retrieved_chunks),
        )

        # 3. Build prompt
        built_prompt = prompt_builder.build(
            query=request.question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=request.conversation_history,
        )

        # 4. LLM inference
        result: InferenceResult = await llm_inference.query(
            built_prompt=built_prompt,
            conversation_history=request.conversation_history,
        )

        # 5. Determine if answer was grounded (not the "not found" response)
        was_grounded = (
            bool(retrieved_chunks)
            and "not found in the course materials" not in result.response_text.lower()
        )

        total_latency = int((time.monotonic() - start_time) * 1000)

        return RAGQueryResponse(
            answer=result.response_text,
            citations=result.citations,
            source_map=result.source_map,
            retrieved_chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "content_preview": c.content[:300],
                    "source_filename": c.source_filename,
                    "document_type": c.document_type,
                    "page_number": c.page_number,
                    "slide_number": c.slide_number,
                    "rerank_score": c.rerank_score,
                }
                for c in retrieved_chunks
            ],
            latency_ms=total_latency,
            model=result.model,
            was_grounded=was_grounded,
            was_compressed=result.was_compressed,
            context_count=len(retrieved_chunks),
        )

    async def stream_query(
        self,
        db: AsyncSession,
        request: RAGQueryRequest,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming RAG query — yields SSE-formatted chunks.
        First yields metadata, then streams tokens, then yields citations.

        SSE event format:
          data: {"type": "token", "content": "..."}
          data: {"type": "citations", "data": [...]}
          data: {"type": "done", "latency_ms": 1234}
        """
        import json

        start_time = time.monotonic()

        query_filter = QueryFilter(
            course=request.course,
            document_types=request.document_types,
        )

        retrieved_chunks = await hybrid_retriever.retrieve(
            session=db,
            query=request.question,
            filters=query_filter,
        )

        built_prompt = prompt_builder.build(
            query=request.question,
            retrieved_chunks=retrieved_chunks,
        )

        # Yield retrieval metadata before streaming
        yield f"data: {json.dumps({'type': 'context', 'count': len(retrieved_chunks)})}\n\n"

        # Stream LLM tokens
        full_response = []
        async for token in llm_inference.stream_query(built_prompt, request.conversation_history):
            full_response.append(token)
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Post-stream: extract citations from complete response
        complete_text = "".join(full_response)
        citations = citation_extractor.extract(
            complete_text,
            built_prompt.source_map,
            built_prompt.context_chunks,
        )

        latency_ms = int((time.monotonic() - start_time) * 1000)

        yield f"data: {json.dumps({'type': 'citations', 'data': citations, 'source_map': {str(k): v for k, v in built_prompt.source_map.items()}})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'latency_ms': latency_ms})}\n\n"


# Singleton
rag_service = RAGService()
