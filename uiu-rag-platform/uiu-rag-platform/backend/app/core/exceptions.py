"""
UIU Academic RAG Platform — Exception Hierarchy
Typed domain exceptions that map cleanly to HTTP status codes.
Enables clean error propagation through the system without coupling
business logic to HTTP concerns.
"""
from __future__ import annotations

from typing import Any


class UIURagException(Exception):
    """Base exception for all platform errors."""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ── Ingestion Exceptions ──────────────────────────────────────────────────────
class IngestionError(UIURagException):
    status_code = 500
    error_code = "INGESTION_ERROR"


class DriveConnectionError(IngestionError):
    error_code = "DRIVE_CONNECTION_ERROR"


class UnsupportedFileTypeError(IngestionError):
    status_code = 422
    error_code = "UNSUPPORTED_FILE_TYPE"


class FileSizeLimitError(IngestionError):
    status_code = 413
    error_code = "FILE_SIZE_LIMIT_EXCEEDED"


class DuplicateDocumentError(IngestionError):
    status_code = 409
    error_code = "DUPLICATE_DOCUMENT"


# ── Parsing Exceptions ────────────────────────────────────────────────────────
class ParseError(UIURagException):
    status_code = 422
    error_code = "PARSE_ERROR"


class OCRError(ParseError):
    error_code = "OCR_ERROR"


class CorruptedFileError(ParseError):
    error_code = "CORRUPTED_FILE"


# ── Embedding Exceptions ──────────────────────────────────────────────────────
class EmbeddingError(UIURagException):
    status_code = 500
    error_code = "EMBEDDING_ERROR"


class EmbeddingModelNotLoadedError(EmbeddingError):
    error_code = "EMBEDDING_MODEL_NOT_LOADED"


# ── Retrieval Exceptions ──────────────────────────────────────────────────────
class RetrievalError(UIURagException):
    status_code = 500
    error_code = "RETRIEVAL_ERROR"


class NoRelevantContextError(RetrievalError):
    status_code = 404
    error_code = "NO_RELEVANT_CONTEXT"


# ── LLM Exceptions ───────────────────────────────────────────────────────────
class LLMError(UIURagException):
    status_code = 502
    error_code = "LLM_ERROR"


class LLMRateLimitError(LLMError):
    status_code = 429
    error_code = "LLM_RATE_LIMIT"


class LLMTimeoutError(LLMError):
    status_code = 504
    error_code = "LLM_TIMEOUT"


# ── Auth Exceptions ───────────────────────────────────────────────────────────
class AuthError(UIURagException):
    status_code = 401
    error_code = "AUTHENTICATION_FAILED"


class PermissionError(UIURagException):
    status_code = 403
    error_code = "PERMISSION_DENIED"


class InvalidTokenError(AuthError):
    error_code = "INVALID_TOKEN"


class ExpiredTokenError(AuthError):
    error_code = "EXPIRED_TOKEN"


class DomainRestrictionError(AuthError):
    status_code = 403
    error_code = "DOMAIN_NOT_ALLOWED"


# ── Resource Exceptions ───────────────────────────────────────────────────────
class NotFoundError(UIURagException):
    status_code = 404
    error_code = "NOT_FOUND"


class ValidationError(UIURagException):
    status_code = 422
    error_code = "VALIDATION_ERROR"
