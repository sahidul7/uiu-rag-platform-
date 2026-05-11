"""Query endpoint with JWT auth + rate limiting"""
from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user
from app.core.rate_limit import check_rate_limit
from app.db.session import get_db
from app.models.document import User
from app.services.rag_service import RAGQueryRequest, rag_service

router = APIRouter()

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    course: str | None = None
    document_types: list[str] | None = None
    semester: str | None = None
    conversation_id: str | None = None
    stream: bool = True

@router.post("/")
async def query_rag(
    request: Request,
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    # Rate limit: 20 queries/minute per user
    check_rate_limit(f"query:{current_user.id}", max_requests=20, window_seconds=60)

    rag_request = RAGQueryRequest(
        question=body.question,
        course=body.course,
        document_types=body.document_types,
        semester=body.semester,
        conversation_id=body.conversation_id,
        stream=body.stream,
    )

    if body.stream:
        return StreamingResponse(
            rag_service.stream_query(db, rag_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    response = await rag_service.query(db, rag_request)
    return response
