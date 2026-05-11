"""
UIU Academic RAG Platform — FastAPI Application Entry Point

Startup sequence:
  1. Configure structured logging
  2. Initialize DB (pgvector, tables, indexes)
  3. Preload embedding model (avoids cold start on first request)
  4. Mount API router

Middleware stack:
  - CORS (configured per environment)
  - Request ID injection (for distributed tracing)
  - Response time header (for SLA monitoring)
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import UIURagException
from app.core.logging import configure_logging, get_logger, set_request_id
from app.db.session import health_check, init_db
from app.embeddings.pipeline import EmbeddingModel
from app.retrieval.hybrid_retriever import BGEReranker

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — runs startup and shutdown logic."""
    logger.info("Starting UIU Academic RAG Platform", version=settings.APP_VERSION)

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Preload ML models
    await EmbeddingModel.load()
    await BGEReranker.load()
    logger.info("ML models preloaded")

    logger.info("Platform ready", environment=settings.ENVIRONMENT)
    yield

    logger.info("Shutting down platform")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade Academic RAG Platform for UIU CSE Department",
    docs_url="/docs" if settings.DEBUG else None,   # Disable Swagger in production
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)


# ── Request ID Middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)
    start_time = time.monotonic()

    response = await call_next(request)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

    logger.info(
        "Request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=elapsed_ms,
    )
    return response


# ── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(UIURagException)
async def handle_domain_exception(request: Request, exc: UIURagException) -> JSONResponse:
    logger.warning(
        "Domain exception",
        error_code=exc.error_code,
        message=exc.message,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unexpected exception", error=str(exc), path=request.url.path, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Please try again.",
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────────
from app.api.v1.router import router as api_v1_router  # noqa: E402

app.include_router(api_v1_router, prefix="/api/v1")


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health() -> dict[str, Any]:
    db_ok = await health_check()
    return {
        "status": "healthy" if db_ok else "degraded",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "database": "ok" if db_ok else "error",
    }
