"""
Request and response models for the FastAPI endpoints.

Separated from documents.py because these models define the *API contract* —
they may differ from internal representations (e.g. we accept a file upload
but internally convert to SourceMetadata + Chunk).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    A query against the indexed document corpus.

    The model enforces minimum query length to prevent empty or trivial searches
    that would return meaningless results and waste API tokens.
    """

    query: str = Field(
        min_length=3,
        max_length=4096,
        description="Natural language query. Minimum 3 characters to avoid trivial searches.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return after reranking.",
    )
    stream: bool = Field(
        default=True,
        description="If true, response is streamed via SSE. If false, returns a single JSON response.",
    )
    include_sources: bool = Field(
        default=True,
        description="If true, the response includes source document citations.",
    )
    filters: dict[str, str | int | None] | None = Field(
        default=None,
        description="Optional OData filter conditions, e.g. {'ticker': 'AAPL', 'fiscal_year': 2025}.",
    )


class ChunkResult(BaseModel):
    """
    A single chunk returned to the user as a citation source.
    """

    chunk_id: str = Field(description="ID of the matching chunk.")
    document_id: str = Field(description="ID of the source document.")
    content: str = Field(description="Excerpt of the chunk content (truncated for display).")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Reranker score — higher is more relevant.",
    )
    filename: str = Field(description="Source filename.")
    ticker: str | None = Field(default=None, description="Associated ticker symbol if applicable.")


class QueryResponse(BaseModel):
    """
    The final API response returned to the caller.

    When stream=True, this is sent as the final metadata frame after the SSE stream ends.
    When stream=False, this is the complete JSON response body.
    """

    answer: str = Field(description="The generated answer to the user's query.")
    sources: list[ChunkResult] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer — ordered by relevance.",
    )
    total_sources_retrieved: int = Field(
        ge=0,
        description="Total candidate chunks retrieved before reranking.",
    )
    processing_time_ms: int = Field(
        ge=0,
        description="End-to-end processing time in milliseconds.",
    )
    degraded_mode: bool = Field(
        default=False,
        description="True if the system fell back to degraded operation (e.g. LLM unavailable).",
    )


class IngestionResponse(BaseModel):
    """
    Response returned after a document ingestion request.
    """

    message: str = Field(description="Human-readable status message.")
    results: list = Field(
        default_factory=list,
        description="List of IndexingResult objects, one per document.",
    )
    total_documents: int = Field(ge=0, description="Number of documents processed.")
    total_chunks_indexed: int = Field(ge=0, description="Total chunks added to the index.")
    processing_time_ms: int = Field(
        ge=0,
        description="Total ingestion time in milliseconds.",
    )


class ErrorResponse(BaseModel):
    """
    Structured error response — consistent across all endpoints.
    No stack traces leak to the client in production.
    """

    error: str = Field(description="Short error code, e.g. 'invalid_document'.")
    message: str = Field(description="Human-readable error detail.")
    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for tracing — include when reporting bugs.",
    )
