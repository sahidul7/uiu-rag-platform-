"""
Documents API — list documents + direct PDF/PPTX/DOCX upload
"""
from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user, get_current_admin
from app.db.session import get_db
from app.models.document import Document, User
from app.services.ingestion_service import ingestion_service
from app.ingestion.parsers.pdf_parser import document_parser
from app.chunking.chunk_manager import chunk_manager
from app.embeddings.pipeline import embedding_pipeline
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.models.document import Document as DocModel, DocumentChunk
import uuid

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

DOCUMENT_TYPES = [
    "lecture_slide","notes","previous_questions",
    "assignment","ct_questions","book","lab_manual"
]

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    course: str = Form(...),
    document_type: str = Form(...),
    semester: str | None = Form(None),
    topic: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF/DOCX/PPTX directly — no Google Drive needed."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Upload PDF, DOCX, or PPTX.",
        )
    if document_type not in DOCUMENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid document_type. Choose from: {DOCUMENT_TYPES}",
        )

    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 100MB.")

    # Process in background
    background_tasks.add_task(
        _process_upload, contents, file.filename or "upload.pdf",
        file.content_type or "application/pdf", course, document_type,
        semester, topic,
    )

    return {"message": "File uploaded and processing started.", "filename": file.filename}


async def _process_upload(
    contents: bytes, filename: str, mime_type: str,
    course: str, document_type: str, semester: str | None, topic: str | None,
):
    """Background task: parse → chunk → embed → store."""
    import asyncio
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / filename
            path.write_bytes(contents)

            parsed = document_parser.parse(path, mime_type)
            chunks = chunk_manager.chunk_document(
                parsed, course=course, document_type=document_type,
                source_filename=filename, semester=semester, topic=topic,
            )
            if not chunks:
                logger.warning("No chunks from upload", filename=filename)
                return

            embeddings = await embedding_pipeline.embed_chunks(
                [c.content for c in chunks],
                [c.content_hash for c in chunks],
            )

            async with AsyncSessionLocal() as db:
                doc = DocModel(
                    drive_file_id=f"upload_{uuid.uuid4().hex}",
                    filename=filename, mime_type=mime_type,
                    file_size_bytes=len(contents), course=course,
                    document_type=document_type, semester=semester,
                    ingestion_status="completed", chunk_count=len(chunks),
                    page_count=parsed.total_pages, has_ocr=parsed.has_ocr,
                )
                db.add(doc)
                await db.flush()

                db.add_all([
                    DocumentChunk(
                        document_id=doc.id, content=c.content,
                        content_hash=c.content_hash, token_count=c.token_count,
                        chunk_index=c.chunk_index, page_number=c.page_number,
                        slide_number=c.slide_number, section_heading=c.section_heading,
                        course=c.course, document_type=c.document_type,
                        source_filename=c.source_filename, topic=c.topic,
                        semester=c.semester, extra_metadata=c.extra_metadata,
                        embedding=emb,
                    )
                    for c, emb in zip(chunks, embeddings)
                ])
                await db.commit()
                logger.info("Upload processed", filename=filename, chunks=len(chunks))
    except Exception as e:
        logger.error("Upload processing failed", filename=filename, error=str(e))


@router.get("/")
async def list_documents(
    course: str | None = Query(None),
    document_type: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    q = select(Document)
    if course: q = q.where(Document.course == course)
    if document_type: q = q.where(Document.document_type == document_type)
    if status: q = q.where(Document.ingestion_status == status)
    q = q.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(q)
    docs = result.scalars().all()
    return [
        {"id": d.id, "filename": d.filename, "course": d.course,
         "document_type": d.document_type, "chunk_count": d.chunk_count,
         "page_count": d.page_count, "ingestion_status": d.ingestion_status,
         "semester": d.semester, "has_ocr": d.has_ocr}
        for d in docs
    ]

@router.get("/courses")
async def list_courses(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document.course).distinct())
    return {"courses": sorted([r[0] for r in result.fetchall() if r[0]])}

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    await db.delete(doc)
    await db.commit()
    return {"message": "Document deleted."}
