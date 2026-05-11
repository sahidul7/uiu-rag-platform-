"""
UIU Academic RAG Platform — ORM Models
Full schema: documents, chunks (with pgvector), users, conversations.
pgvector Column type enables native cosine similarity queries.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


# ── User ──────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=generate_uuid
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    hashed_password: Mapped[str | None] = mapped_column(String(255))
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")


# ── Document ──────────────────────────────────────────────────────────────────
class Document(Base):
    """
    Represents an ingested source file.
    drive_file_id enables deduplication and incremental sync.
    """
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    drive_file_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str | None] = mapped_column(Text)          # local cache path
    mime_type: Mapped[str] = mapped_column(String(128))
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    drive_modified_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Academic metadata
    course: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # document_type: lecture_slide | notes | previous_questions | assignment
    #                ct_questions | book | lab_manual
    semester: Mapped[str | None] = mapped_column(String(50))
    topic: Mapped[str | None] = mapped_column(String(255))
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Ingestion state
    ingestion_status: Mapped[str] = mapped_column(
        String(50), default="pending", index=True
    )  # pending | processing | completed | failed
    ingestion_error: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int | None] = mapped_column(Integer)
    has_ocr: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_documents_course_type", "course", "document_type"),
        Index("idx_documents_status", "ingestion_status"),
    )


# ── DocumentChunk ─────────────────────────────────────────────────────────────
class DocumentChunk(Base):
    """
    Core retrieval unit. Each chunk carries:
    - semantic content
    - pgvector embedding (384-dim by default)
    - rich academic metadata for filtered retrieval
    - BM25-friendly tsvector (computed in DB)
    """
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    # Position
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)   # 0-based position in document
    page_number: Mapped[int | None] = mapped_column(Integer)
    slide_number: Mapped[int | None] = mapped_column(Integer)
    section_heading: Mapped[str | None] = mapped_column(Text)

    # Academic metadata (denormalized from Document for query efficiency)
    course: Mapped[str] = mapped_column(String(100), nullable=False)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False)
    source_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    topic: Mapped[str | None] = mapped_column(String(255))
    semester: Mapped[str | None] = mapped_column(String(50))
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Vector embedding — dimension must match EMBEDDING_DIMENSION in config
    embedding: Mapped[list[float]] = mapped_column(Vector(384))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_course", "course"),
        Index("idx_chunks_course_type", "course", "document_type"),
        Index("idx_chunks_content_hash", "content_hash"),
        # HNSW vector index created separately in init_db() — DDL constraint
    )


# ── Conversation ──────────────────────────────────────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str | None] = mapped_column(String(512))
    course: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


# ── Message ───────────────────────────────────────────────────────────────────
class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # user | assistant | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


# ── IngestionJob ──────────────────────────────────────────────────────────────
class IngestionJob(Base):
    """Tracks background ingestion jobs for observability and retry."""
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=generate_uuid)
    job_type: Mapped[str] = mapped_column(String(50))   # full_sync | incremental_sync | single_file
    status: Mapped[str] = mapped_column(String(50), default="pending")
    documents_discovered: Mapped[int] = mapped_column(Integer, default=0)
    documents_processed: Mapped[int] = mapped_column(Integer, default=0)
    documents_failed: Mapped[int] = mapped_column(Integer, default=0)
    error_log: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
