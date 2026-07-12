"""
Structured error handling with typed exceptions and consistent API error responses.

Design decisions:
  - Custom exception classes carry structured data (status code, detail, correlation ID)
  - FastAPI exception handlers convert them to consistent JSON responses
  - No stack traces leak to the client in production
  - Correlation IDs link errors to logs and traces for debugging

Graceful degradation:
  Instead of returning 500 for every failure, we categorise errors:
    - 4xx = client errors (bad request, not found)
    - 503 = service unavailable (Azure downstream, degraded mode)
    - 500 = unexpected internal errors (bugs)
"""

from __future__ import annotations

import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models.requests import ErrorResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "internal_error",
        correlation_id: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.correlation_id = correlation_id or str(uuid.uuid4())
        super().__init__(self.message)


class DocumentParseError(RAGError):
    """Raised when document parsing fails (unsupported format, corrupted file, etc.)."""

    def __init__(self, message: str = "Failed to parse document", correlation_id: str | None = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="document_parse_error",
            correlation_id=correlation_id,
        )


class EmbeddingError(RAGError):
    """Raised when embedding generation fails and no fallback is available."""

    def __init__(self, message: str = "Embedding service unavailable", correlation_id: str | None = None):
        super().__init__(
            message=message,
            status_code=503,
            error_code="embedding_unavailable",
            correlation_id=correlation_id,
        )


class SearchIndexError(RAGError):
    """Raised when Azure AI Search is unreachable or returns an error."""

    def __init__(self, message: str = "Search index unavailable", correlation_id: str | None = None):
        super().__init__(
            message=message,
            status_code=503,
            error_code="search_index_unavailable",
            correlation_id=correlation_id,
        )


class LLMGenerationError(RAGError):
    """Raised when the LLM generation step fails."""

    def __init__(self, message: str = "LLM generation service unavailable", correlation_id: str | None = None):
        super().__init__(
            message=message,
            status_code=503,
            error_code="llm_generation_unavailable",
            correlation_id=correlation_id,
        )


# ── FastAPI Exception Handlers ──────────────────────────────────────────────


async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
    """Handle custom RAGError exceptions."""
    logger.warning(
        "RAGError handled",
        error_code=exc.error_code,
        status=exc.status_code,
        correlation_id=exc.correlation_id,
        message=exc.message,
    )
    response = ErrorResponse(
        error=exc.error_code,
        message=exc.message,
        correlation_id=exc.correlation_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors — return 400 with field-level details."""
    correlation_id = str(uuid.uuid4())
    errors = exc.errors()
    detail = "; ".join(f"{e['loc']}: {e['msg']}" for e in errors)

    logger.warning(
        "Validation error",
        correlation_id=correlation_id,
        errors=errors,
    )
    response = ErrorResponse(
        error="validation_error",
        message=detail,
        correlation_id=correlation_id,
    )
    return JSONResponse(
        status_code=400,
        content=response.model_dump(),
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected errors — returns 500 without leaking details."""
    correlation_id = str(uuid.uuid4())
    logger.error(
        "Unhandled exception",
        correlation_id=correlation_id,
        error=str(exc),
        exc_info=True,
    )
    response = ErrorResponse(
        error="internal_error",
        message="An unexpected error occurred. Please retry with the correlation ID.",
        correlation_id=correlation_id,
    )
    return JSONResponse(
        status_code=500,
        content=response.model_dump(),
    )


def register_error_handlers(app):
    """Register all error handlers with a FastAPI application."""
    app.add_exception_handler(RAGError, rag_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)
