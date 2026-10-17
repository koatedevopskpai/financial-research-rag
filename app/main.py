"""
FastAPI application entry point for Financial Research RAG.

Exposes two primary endpoints:
  POST /ingest    — Upload financial documents for chunking, embedding, and indexing
  POST /query     — Ask questions against the indexed documents (supports SSE streaming)

Architecture decision — single FastAPI app, not a microservice split:
  For the query volume of an investment research team (not internet-scale),
  a single deployable unit reduces operational complexity, keeps query latency
  low (no inter-service hops), and makes end-to-end tracing straightforward.
  If scale demands it, ingestion and query can be split later.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.models.requests import QueryRequest, IngestionResponse, ErrorResponse
from app.pipelines.document_pipeline import DocumentPipeline
from app.pipelines.query_pipeline import QueryPipeline
from app.utils.logging import get_logger
from app.utils.monitoring import setup_observability

logger = get_logger(__name__)

# ── Global pipeline instances ─────────────────────────────────────────────────
# Initialised once at startup and reused across requests.
# This avoids re-creating Azure SDK clients on every request (expensive handshake).
document_pipeline: DocumentPipeline | None = None
query_pipeline: QueryPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — replaces the deprecated on_event("startup") / on_event("shutdown").

    Initialises:
      1. OpenTelemetry tracing (if configured)
      2. Document ingestion pipeline
      3. Query pipeline (retrieval + generation)

    Graceful degradation: if Azure is unreachable at startup, pipelines initialise
    in degraded mode and log a warning rather than crashing the app.
    """
    global document_pipeline, query_pipeline

    setup_observability()
    logger.info("Initialising pipelines...", service=settings.otel_service_name)

    try:
        document_pipeline = DocumentPipeline()
        query_pipeline = QueryPipeline(document_pipeline.search_index)
        logger.info("Pipelines initialised successfully")
    except Exception as exc:
        logger.warning(
            "Pipelines initialised in degraded mode — Azure services may be unavailable",
            error=str(exc),
        )
    yield


app = FastAPI(
    title="Financial Research RAG",
    version="0.1.0",
    description="Production-grade RAG system for financial investment research. "
    "Ingest documents and ask natural-language questions against them.",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# In production, restrict origins to the specific frontend domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health_check():
    """
    Kubernetes liveness/readiness probe.

    Returns the pipeline initialisation status so K8s can restart the pod
    if Azure connectivity is permanently lost.
    """
    return {
        "status": "ok",
        "document_pipeline_ready": document_pipeline is not None,
        "query_pipeline_ready": query_pipeline is not None,
    }


# ── Ingestion ──────────────────────────────────────────────────────────────────


@app.post(
    "/ingest",
    tags=["Ingestion"],
    response_model=IngestionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a financial document for chunking, embedding, and indexing.

    Supported file types: PDF, DOCX, XLSX, TXT, HTML
    Max file size: determined by infrastructure (configure via reverse proxy / API gateway).

    The ingestion pipeline:
      1. Parses the document into raw text
      2. Applies hierarchical chunking (parent + child chunks)
      3. Generates embeddings via Azure OpenAI
      4. Pushes chunks + embeddings to Azure AI Search

    Returns immediately with indexing results — documents are searchable within seconds.
    """
    if document_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Ingestion pipeline not initialised. Check Azure connectivity.",
        )

    start = time.monotonic()

    # Validate file type before processing
    allowed_types = {"pdf", "docx", "xlsx", "txt", "html"}
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(allowed_types)}",
        )

    contents = await file.read()
    result = await document_pipeline.process_document(
        contents=contents,
        filename=file.filename,
        file_type=ext,
    )

    elapsed_ms = result.processing_time_ms or int((time.monotonic() - start) * 1000)

    logger.info(
        "Document ingested",
        filename=file.filename,
        chunks=result.total_chunks,
        elapsed_ms=elapsed_ms,
    )
    return IngestionResponse(
        message=(
            "Document indexed successfully"
            if result.success
            else f"Ingestion failed: {result.error_message}"
        ),
        results=[result],
        total_documents=1,
        total_chunks_indexed=result.total_chunks,
        processing_time_ms=elapsed_ms,
    )


# ── Query ──────────────────────────────────────────────────────────────────────


@app.post(
    "/query",
    tags=["Query"],
    responses={503: {"model": ErrorResponse}},
)
async def query_documents(request: QueryRequest):
    """
    Ask a question against the indexed documents.

    When stream=True (default), the response is an SSE event stream:
      event: token
      data: {"token": "The"}

      event: token
      data: {"token": "EBITDA"}

      ...

      event: done
      data: {"answer": "...", "sources": [...], "processing_time_ms": 1234}

    When stream=False, returns the complete QueryResponse as JSON.

    Graceful degradation:
      - If Azure OpenAI is unavailable, returns a keyword-based extractive answer
      - If Azure AI Search is unavailable, returns an error with degraded_mode=True
    """
    if query_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Query pipeline not initialised. Check Azure connectivity.",
        )

    start = time.monotonic()

    if request.stream:
        # Return an SSE stream — the pipeline yields tokens as they arrive.
        return EventSourceResponse(
            query_pipeline.stream_query(request),
            media_type="text/event-stream",
        )

    # Non-streaming path: collect the full response, then return JSON.
    response = await query_pipeline.query(request)
    response.processing_time_ms = int((time.monotonic() - start) * 1000)
    return response
