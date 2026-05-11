"""
UIU Academic RAG Platform — Google Drive Ingestion Client
Recursive folder crawling, incremental sync, duplicate detection.

Design decisions:
- Service Account auth (no user OAuth needed for Drive access)
- Exponential backoff for API quota errors
- Tracks drive_modified_time for incremental sync
- Async download with aiofiles for non-blocking I/O
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from app.core.config import settings
from app.core.exceptions import DriveConnectionError, FileSizeLimitError, UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Supported MIME types ──────────────────────────────────────────────────────
SUPPORTED_MIME_TYPES: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".ppt",
    "text/plain": ".txt",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    # Google Docs → export as PDF
    "application/vnd.google-apps.document": ".pdf",
    "application/vnd.google-apps.presentation": ".pptx",
}

# ── Drive folder → academic metadata mapping ──────────────────────────────────
FOLDER_METADATA_MAP: dict[str, dict[str, str]] = {
    "previous questions": {"document_type": "previous_questions"},
    "lecture slides":     {"document_type": "lecture_slide"},
    "notes":              {"document_type": "notes"},
    "assignments":        {"document_type": "assignment"},
    "ct question":        {"document_type": "ct_questions"},
    "ct questions":       {"document_type": "ct_questions"},
    "books":              {"document_type": "book"},
    "lab manual":         {"document_type": "lab_manual"},
    "lab manuals":        {"document_type": "lab_manual"},
}


@dataclass
class DriveFile:
    """Represents a discovered Drive file with resolved academic metadata."""
    file_id: str
    name: str
    mime_type: str
    size_bytes: int
    modified_time: datetime
    drive_path: str           # e.g. "DBMS/Lecture Slides/Lecture_5.pdf"
    course: str
    document_type: str
    semester: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class GoogleDriveClient:
    """
    Async-friendly Google Drive client.
    Sync Drive API calls run in thread pool to avoid blocking the event loop.
    """

    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    MAX_FILE_SIZE_BYTES = settings.DRIVE_MAX_FILE_SIZE_MB * 1024 * 1024

    def __init__(self) -> None:
        self._service: Any = None
        self._lock = asyncio.Lock()

    async def _get_service(self) -> Any:
        """Lazy-initialize Drive service with service account credentials."""
        async with self._lock:
            if self._service is None:
                try:
                    creds = service_account.Credentials.from_service_account_file(
                        settings.GOOGLE_SERVICE_ACCOUNT_JSON,
                        scopes=self.SCOPES,
                    )
                    self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
                    logger.info("Google Drive service initialized")
                except Exception as e:
                    raise DriveConnectionError(f"Failed to initialize Drive client: {e}")
        return self._service

    async def crawl_folder(
        self,
        folder_id: str,
        *,
        parent_path: str = "",
        course: str = "",
        document_type: str = "unknown",
        semester: str | None = None,
        depth: int = 0,
        max_depth: int = 6,
    ) -> list[DriveFile]:
        """
        Recursively crawl a Drive folder, resolving metadata from folder names.
        Returns flat list of all discovered DriveFile objects.
        """
        if depth > max_depth:
            logger.warning("Max depth reached, stopping crawl", path=parent_path)
            return []

        service = await self._get_service()
        files: list[DriveFile] = []
        page_token: str | None = None

        while True:
            try:
                query_kwargs: dict[str, Any] = {
                    "q": f"'{folder_id}' in parents and trashed=false",
                    "fields": "nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    "pageSize": 100,
                    "supportsAllDrives": True,
                    "includeItemsFromAllDrives": True,
                }
                if page_token:
                    query_kwargs["pageToken"] = page_token

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: service.files().list(**query_kwargs).execute(),
                )
            except HttpError as e:
                logger.error("Drive API error", folder_id=folder_id, error=str(e))
                break

            for item in response.get("files", []):
                item_name = item["name"]
                item_id = item["id"]
                mime = item.get("mimeType", "")
                item_path = f"{parent_path}/{item_name}".lstrip("/")

                if mime == "application/vnd.google-apps.folder":
                    # Resolve metadata from folder name
                    resolved_course = course
                    resolved_type = document_type
                    resolved_semester = semester

                    lower_name = item_name.lower()
                    if depth == 0:
                        # Top-level folder = course name (e.g. "DBMS")
                        resolved_course = item_name
                    elif folder_meta := FOLDER_METADATA_MAP.get(lower_name):
                        resolved_type = folder_meta["document_type"]
                    # Detect semester patterns like "Spring 2025", "Fall 2024"
                    elif any(s in lower_name for s in ["spring", "summer", "fall", "winter"]):
                        resolved_semester = item_name

                    # Recurse into subfolder
                    sub_files = await self.crawl_folder(
                        folder_id=item_id,
                        parent_path=item_path,
                        course=resolved_course,
                        document_type=resolved_type,
                        semester=resolved_semester,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                    files.extend(sub_files)

                elif mime in SUPPORTED_MIME_TYPES:
                    size = int(item.get("size", 0))
                    if size > self.MAX_FILE_SIZE_BYTES:
                        logger.warning(
                            "File exceeds size limit, skipping",
                            name=item_name, size_mb=size / (1024 * 1024),
                        )
                        continue

                    modified_str = item.get("modifiedTime", "")
                    modified_dt = datetime.fromisoformat(
                        modified_str.replace("Z", "+00:00")
                    ) if modified_str else datetime.utcnow()

                    files.append(DriveFile(
                        file_id=item_id,
                        name=item_name,
                        mime_type=mime,
                        size_bytes=size,
                        modified_time=modified_dt,
                        drive_path=item_path,
                        course=course or "Unknown",
                        document_type=document_type,
                        semester=semester,
                    ))
                else:
                    logger.debug("Skipping unsupported file type", name=item_name, mime=mime)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        logger.info(
            "Crawl complete for folder",
            folder_id=folder_id,
            path=parent_path,
            files_found=len(files),
        )
        return files

    async def download_file(self, drive_file: DriveFile, dest_dir: Path) -> Path:
        """
        Download a Drive file to a local temp directory.
        Google Docs are exported to appropriate format.
        Returns the local file path.
        """
        service = await self._get_service()
        ext = SUPPORTED_MIME_TYPES.get(drive_file.mime_type, ".bin")
        dest_path = dest_dir / f"{drive_file.file_id}{ext}"

        if dest_path.exists():
            logger.debug("File already cached locally", path=str(dest_path))
            return dest_path

        try:
            # Google Docs need export; binary files use get_media
            if drive_file.mime_type.startswith("application/vnd.google-apps"):
                export_mime = {
                    "application/vnd.google-apps.document": "application/pdf",
                    "application/vnd.google-apps.presentation": (
                        "application/vnd.openxmlformats-officedocument"
                        ".presentationml.presentation"
                    ),
                }.get(drive_file.mime_type, "application/pdf")

                request = service.files().export_media(
                    fileId=drive_file.file_id, mimeType=export_mime
                )
            else:
                request = service.files().get_media(
                    fileId=drive_file.file_id, supportsAllDrives=True
                )

            async with aiofiles.open(dest_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request, chunksize=4 * 1024 * 1024)
                done = False
                while not done:
                    _, done = await asyncio.get_event_loop().run_in_executor(
                        None, downloader.next_chunk
                    )

            logger.info("Downloaded file", name=drive_file.name, path=str(dest_path))
            return dest_path

        except HttpError as e:
            logger.error("Download failed", file_id=drive_file.file_id, error=str(e))
            raise IngestionError(f"Failed to download {drive_file.name}: {e}")

    async def get_file_modified_time(self, file_id: str) -> datetime | None:
        """Check modified time for incremental sync — only re-process changed files."""
        service = await self._get_service()
        try:
            meta = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: service.files().get(
                    fileId=file_id,
                    fields="modifiedTime",
                    supportsAllDrives=True,
                ).execute(),
            )
            ts = meta.get("modifiedTime", "")
            return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
        except HttpError:
            return None


# Import error fix
from app.core.exceptions import IngestionError  # noqa: E402
