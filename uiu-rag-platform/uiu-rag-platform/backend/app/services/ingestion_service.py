"""
UIU Academic RAG Platform — Ingestion Service
Orchestrates the full ingestion pipeline:
  Google Drive → Download → Parse → Chunk → Embed → Store

Features:
  - Incremental sync (skip unchanged files via drive_modified_time)
  - Duplicate detection (content hash + drive_file_id)
  - Async pipeline with controlled concurrency
  - Per-file error isolation (one failure doesn't stop the batch)
  - Job tracking for observability
"""
from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking.chunk_manager import Chunk, chunk_manager
from app.core.config import settings
from app.core.exceptions import ParseError, IngestionError
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.embeddings.pipeline import embedding_pipeline
from app.ingestion.drive_client import DriveFile, GoogleDriveClient
from app.ingestion.parsers.pdf_parser import document_parser
from app.models.document import Document, DocumentChunk, IngestionJob

logger = get_logger(__name__)

# Limit concurrent file processing to avoid overwhelming DB/embedding model
CONCURRENCY_LIMIT = 3


class IngestionService:
    def __init__(self) -> None:
        self._drive_client = GoogleDriveClient()
        self._semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def run_full_sync(self) -> str:
        """
        Full ingestion from Google Drive root.
        Creates an IngestionJob for tracking.
        Returns the job_id.
        """
        async with AsyncSessionLocal() as db:
            job = IngestionJob(
                job_type="full_sync",
                status="running",
                started_at=datetime.now(timezone.utc),
            )
            db.add(job)
            await db.commit()
            await db.refresh(job)
            job_id = job.id

        logger.info("Starting full sync", job_id=job_id)

        try:
            # Discover all files recursively
            all_files = await self._drive_client.crawl_folder(
                folder_id=settings.GOOGLE_DRIVE_ROOT_FOLDER_ID,
            )

            logger.info("Discovery complete", total_files=len(all_files), job_id=job_id)

            # Process in controlled concurrency
            results = await self._process_files_concurrently(all_files, job_id)

            succeeded = sum(1 for r in results if r)
            failed = sum(1 for r in results if not r)

            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == job_id)
                    .values(
                        status="completed",
                        documents_discovered=len(all_files),
                        documents_processed=succeeded,
                        documents_failed=failed,
                        completed_at=datetime.now(timezone.utc),
                    )
                )
                await db.commit()

            logger.info(
                "Full sync complete",
                job_id=job_id,
                succeeded=succeeded,
                failed=failed,
            )
        except Exception as e:
            logger.error("Full sync failed", job_id=job_id, error=str(e))
            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == job_id)
                    .values(status="failed", completed_at=datetime.now(timezone.utc))
                )
                await db.commit()

        return job_id

    async def run_incremental_sync(self) -> str:
        """Only re-process files modified since last ingestion."""
        async with AsyncSessionLocal() as db:
            job = IngestionJob(
                job_type="incremental_sync",
                status="running",
                started_at=datetime.now(timezone.utc),
            )
            db.add(job)
            await db.commit()
            await db.refresh(job)
            job_id = job.id

        all_files = await self._drive_client.crawl_folder(
            folder_id=settings.GOOGLE_DRIVE_ROOT_FOLDER_ID,
        )

        # Filter to changed/new files
        to_process = []
        async with AsyncSessionLocal() as db:
            for drive_file in all_files:
                existing = await self._get_existing_document(db, drive_file.file_id)
                if existing is None:
                    to_process.append(drive_file)  # New file
                elif (
                    existing.drive_modified_time is None
                    or drive_file.modified_time > existing.drive_modified_time.replace(tzinfo=timezone.utc)
                ):
                    to_process.append(drive_file)  # Modified file

        logger.info(
            "Incremental sync",
            total_discovered=len(all_files),
            to_process=len(to_process),
            job_id=job_id,
        )

        results = await self._process_files_concurrently(to_process, job_id)
        succeeded = sum(1 for r in results if r)
        failed = len(results) - succeeded

        async with AsyncSessionLocal() as db:
            await db.execute(
                update(IngestionJob)
                .where(IngestionJob.id == job_id)
                .values(
                    status="completed",
                    documents_discovered=len(all_files),
                    documents_processed=succeeded,
                    documents_failed=failed,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()

        return job_id

    async def _process_files_concurrently(
        self, files: list[DriveFile], job_id: str
    ) -> list[bool]:
        """Process files with bounded concurrency."""
        tasks = [self._process_single_file(f, job_id) for f in files]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    async def _process_single_file(self, drive_file: DriveFile, job_id: str) -> bool:
        """
        Process one file: download → parse → chunk → embed → store.
        Errors are caught and logged; never propagates to cancel other files.
        """
        async with self._semaphore:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Download
                    local_path = await self._drive_client.download_file(
                        drive_file, Path(tmp_dir)
                    )

                    # Parse
                    parsed_doc = document_parser.parse(local_path, drive_file.mime_type)

                    # Chunk
                    chunks = chunk_manager.chunk_document(
                        parsed_doc,
                        course=drive_file.course,
                        document_type=drive_file.document_type,
                        source_filename=drive_file.name,
                        semester=drive_file.semester,
                    )

                    if not chunks:
                        logger.warning("No chunks produced", file=drive_file.name)
                        return False

                    # Embed
                    contents = [c.content for c in chunks]
                    hashes = [c.content_hash for c in chunks]
                    embeddings = await embedding_pipeline.embed_chunks(contents, hashes)

                    # Store
                    await self._store_document_and_chunks(
                        drive_file, parsed_doc, chunks, embeddings
                    )

                    logger.info(
                        "File ingested",
                        file=drive_file.name,
                        chunks=len(chunks),
                        pages=parsed_doc.total_pages,
                    )
                    return True

            except Exception as e:
                logger.error(
                    "File ingestion failed",
                    file=drive_file.name,
                    error=str(e),
                    job_id=job_id,
                )
                await self._mark_document_failed(drive_file, str(e))
                return False

    async def _store_document_and_chunks(
        self,
        drive_file: DriveFile,
        parsed_doc: Any,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert document + chunks atomically."""
        async with AsyncSessionLocal() as db:
            # Upsert document
            existing = await self._get_existing_document(db, drive_file.file_id)

            if existing:
                # Delete old chunks before reinserting
                from sqlalchemy import delete
                await db.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == existing.id)
                )
                doc = existing
                doc.ingestion_status = "completed"
                doc.drive_modified_time = drive_file.modified_time
                doc.chunk_count = len(chunks)
                doc.page_count = parsed_doc.total_pages
                doc.has_ocr = parsed_doc.has_ocr
            else:
                doc = Document(
                    drive_file_id=drive_file.file_id,
                    filename=drive_file.name,
                    mime_type=drive_file.mime_type,
                    file_size_bytes=drive_file.size_bytes,
                    drive_modified_time=drive_file.modified_time,
                    course=drive_file.course,
                    document_type=drive_file.document_type,
                    semester=drive_file.semester,
                    ingestion_status="completed",
                    chunk_count=len(chunks),
                    page_count=parsed_doc.total_pages,
                    has_ocr=parsed_doc.has_ocr,
                )
                db.add(doc)
                await db.flush()  # get doc.id

            # Insert chunks with embeddings
            chunk_models = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_models.append(DocumentChunk(
                    document_id=doc.id,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    token_count=chunk.token_count,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    slide_number=chunk.slide_number,
                    section_heading=chunk.section_heading,
                    course=chunk.course,
                    document_type=chunk.document_type,
                    source_filename=chunk.source_filename,
                    topic=chunk.topic,
                    semester=chunk.semester,
                    extra_metadata=chunk.extra_metadata,
                    embedding=embedding,
                ))

            db.add_all(chunk_models)
            await db.commit()

    async def _get_existing_document(
        self, db: AsyncSession, drive_file_id: str
    ) -> Document | None:
        result = await db.execute(
            select(Document).where(Document.drive_file_id == drive_file_id)
        )
        return result.scalar_one_or_none()

    async def _mark_document_failed(
        self, drive_file: DriveFile, error: str
    ) -> None:
        async with AsyncSessionLocal() as db:
            existing = await self._get_existing_document(db, drive_file.file_id)
            if existing:
                existing.ingestion_status = "failed"
                existing.ingestion_error = error[:2000]
            else:
                db.add(Document(
                    drive_file_id=drive_file.file_id,
                    filename=drive_file.name,
                    mime_type=drive_file.mime_type,
                    file_size_bytes=drive_file.size_bytes,
                    course=drive_file.course,
                    document_type=drive_file.document_type,
                    ingestion_status="failed",
                    ingestion_error=error[:2000],
                ))
            await db.commit()


# Singleton
ingestion_service = IngestionService()
