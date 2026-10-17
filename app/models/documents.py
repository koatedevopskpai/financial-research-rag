"""
Pydantic models for documents, chunks, and sources that flow through the RAG pipeline.

Every data structure that crosses a service boundary (ingestion → index → retrieval → generation)
is validated by Pydantic v2 *strict* mode. This is an opinionated choice:
  - Strict mode rejects type coercion (no str→int surprises)
  - Fail fast at the boundary, not silently halfway through a pipeline
  - Azure AI Search needs explicit field types — these models drive the index schema
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    """
    Metadata about the original document — preserved across the chunking and indexing pipeline.
    Used for provenance tracking, filtering, and compliance (e.g. regulatory timeboxes).
    """

    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the source document — used as the partition key in AI Search.",
    )
    filename: str = Field(description="Original filename including extension.")
    file_type: Literal["pdf", "docx", "xlsx", "txt", "html"] = Field(
        description="File type determines the parser used in ingestion.",
    )
    upload_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of when the document was ingested.",
    )
    page_count: int | None = Field(
        default=None,
        ge=1,
        description="Number of pages (available for PDF/DOCX after parsing).",
    )
    source_url: str | None = Field(
        default=None,
        description="Original source URL if the document was fetched from a web source.",
    )
    ticker: str | None = Field(
        default=None,
        description="Stock ticker symbol if the document pertains to a specific company.",
    )
    fiscal_year: int | None = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Fiscal year of the report — used for time-filtered queries.",
    )
    document_type: Literal[
        "annual_report", "earnings_transcript", "analyst_note", "regulatory_filing", "other"
    ] = Field(
        default="other",
        description="Classification of the document type — affects chunking strategy.",
    )


class Chunk(BaseModel):
    """
    A single chunk of text extracted from a source document.

    Chunks carry a hierarchical relationship:
      - parent_chunk_id links small chunks to their larger parent for context retrieval
      - metadata is duplicated per chunk to make AI Search filtering efficient
        (avoiding cross-document joins at query time)
    """

    chunk_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this chunk — used as the document key in AI Search.",
    )
    parent_chunk_id: str | None = Field(
        default=None,
        description="ID of the parent (large) chunk — None for parent chunks themselves.",
    )
    chunk_type: Literal["parent", "child"] = Field(
        description="'parent' = large context chunk, 'child' = small retrieval chunk.",
    )
    content: str = Field(description="The text content of this chunk — UTF-8 encoded.")
    token_count: int = Field(
        ge=1,
        description="Exact token count (via tiktoken) — used for LLM context window budgeting.",
    )
    chunk_index: int = Field(
        ge=0,
        description="Zero-based index of this chunk within the document — preserves ordering.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding of this chunk — populated by the embedder service. None before indexing.",
    )
    embedding_dimensions: int | None = Field(
        default=None,
        ge=1,
        description="Number of embedding dimensions — validated against the model's output.",
    )

    # ── Denormalised source metadata (copied from SourceMetadata for search efficiency) ──
    document_id: str = Field(description="Foreign key — SourceMetadata.document_id.")
    filename: str = Field(description="Denormalised — copied from SourceMetadata for filtering.")
    ticker: str | None = Field(default=None, description="Denormalised — copied from SourceMetadata.")
    fiscal_year: int | None = Field(default=None, description="Denormalised — copied from SourceMetadata.")
    document_type: str = Field(
        default="other",
        description="Denormalised — copied from SourceMetadata for filtering.",
    )


class IndexingResult(BaseModel):
    """
    Summary of an indexing operation — returned to the caller after ingestion completes.
    """

    document_id: str = Field(description="ID of the indexed document.")
    filename: str = Field(description="Original filename.")
    parent_chunks_count: int = Field(ge=0, description="Number of parent chunks indexed.")
    child_chunks_count: int = Field(ge=0, description="Number of child chunks indexed.")
    total_chunks: int = Field(ge=0, description="Total chunks (parent + child) indexed.")
    processing_time_ms: int | None = Field(
        default=None,
        ge=0,
        description="Wall-clock duration of the ingest pipeline. Set by the pipeline; None if unmeasured.",
    )
    success: bool = Field(description="True if all chunks were indexed without error.")
    error_message: str | None = Field(
        default=None,
        description="Error details if partial or full failure occurred.",
    )
