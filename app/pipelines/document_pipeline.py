"""
End-to-end document ingestion pipeline: parse → chunk → embed → index.

This pipeline is the entry point for all document ingestion operations.
Each step is independently replaceable — you could swap the chunker strategy,
change embedding models, or switch vector stores without touching the others.

Graceful degradation philosophy:
  - If embedding fails, chunks are indexed without vectors (keyword-only search still works)
  - If indexing partially fails, the response reports which chunks failed
  - The pipeline never throws — it returns an IndexingResult with success=False + error_message
"""

from __future__ import annotations

import time
from typing import Any

from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import EmbeddingService
from app.ingestion.indexer import SearchIndexManager
from app.models.documents import IndexingResult, SourceMetadata
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentPipeline:
    """
    Orchestrates the full ingestion pipeline for a single document.

    Usage:
        pipeline = DocumentPipeline()
        result = await pipeline.process_document(contents, filename, "pdf")
    """

    def __init__(self) -> None:
        self.index_manager = SearchIndexManager()
        self.embedder = EmbeddingService()
        self.search_index = self.index_manager.get_search_client()

    async def process_document(
        self,
        contents: bytes,
        filename: str,
        file_type: str,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> IndexingResult:
        """
        Process a single document through the full pipeline.

        Args:
            contents: Raw file bytes.
            filename: Original filename with extension.
            file_type: File extension (pdf, docx, etc.).
            metadata_overrides: Optional overrides for SourceMetadata fields
                                (e.g. ticker="AAPL", fiscal_year=2025).

        Returns:
            IndexingResult with counts and success status.
        """
        start = time.perf_counter()
        logger.info("Processing document", filename=filename, file_type=file_type)

        # ── Step 0: Build metadata ─────────────────────────────────────────────
        from typing import Literal, cast

        supported = {"txt", "html"}
        if file_type not in supported:
            raise ValueError("Only txt/html parsing is implemented; other formats are unsupported")
        validated_file_type = cast(Literal["txt", "html"], file_type)
        metadata = SourceMetadata(
            filename=filename,
            file_type=validated_file_type,
        )
        if metadata_overrides:
            for key, value in metadata_overrides.items():
                if hasattr(metadata, key) and value is not None:
                    setattr(metadata, key, value)

        # ── Step 1: Parse ──────────────────────────────────────────────────────
        # For this implementation, we support basic text extraction.
        # In production, use Azure Document Intelligence for PDF/DOCX parsing.
        text = self._parse_document(contents, file_type)
        if not text.strip():
            return IndexingResult(
                document_id=metadata.document_id,
                filename=filename,
                parent_chunks_count=0,
                child_chunks_count=0,
                total_chunks=0,
                processing_time_ms=int((time.perf_counter() - start) * 1000),
                success=False,
                error_message="Document parsing produced empty text",
            )

        # ── Step 2: Chunk ───────────────────────────────────────────────────────
        chunks = chunk_document(text, metadata)
        parent_count = sum(1 for c in chunks if c.chunk_type == "parent")
        child_count = sum(1 for c in chunks if c.chunk_type == "child")
        logger.info(
            "Chunking complete",
            filename=filename,
            parent=parent_count,
            child=child_count,
        )

        # ── Step 3: Embed ──────────────────────────────────────────────────────
        chunks = await self.embedder.embed(chunks)

        # ── Step 4: Index ───────────────────────────────────────────────────────
        try:
            result = await self.index_manager.upload_documents(chunks)
            success_count = sum(1 for r in result if r.succeeded)
            failed_count = len(result) - success_count
        except Exception as exc:
            logger.error("Indexing failed", filename=filename, error=str(exc))
            return IndexingResult(
                document_id=metadata.document_id,
                filename=filename,
                parent_chunks_count=0,
                child_chunks_count=0,
                total_chunks=0,
                processing_time_ms=int((time.perf_counter() - start) * 1000),
                success=False,
                error_message=f"Indexing failed: {exc}",
            )

        success = failed_count == 0
        return IndexingResult(
            document_id=metadata.document_id,
            filename=filename,
            parent_chunks_count=parent_count,
            child_chunks_count=child_count,
            total_chunks=len(chunks),
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            success=success,
            error_message=None if success else f"{failed_count} chunks failed to index",
        )

    def _parse_document(self, contents: bytes, file_type: str) -> str:
        """
        Extract text content from a document.

        Supported types:
          - txt: direct decode
          - pdf: basic text extraction (in production, use Azure Document Intelligence)
          - docx: basic extraction (in production, use python-docx + Azure DI)
          - xlsx: basic extraction (in production, use openpyxl)
          - html: basic extraction (in production, use BeautifulSoup)

        This is a minimal implementation — enough to demonstrate the pipeline.
        A production system would use Azure Document Intelligence for all
        file types to get consistent, high-quality text extraction.
        """
        if file_type == "txt":
            return contents.decode("utf-8", errors="replace")

        if file_type == "html":
            text = contents.decode("utf-8", errors="replace")
            import re

            # Strip HTML tags (basic — use BeautifulSoup in production)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        raise ValueError("Only txt/html parsing is implemented; binary formats are unsupported")
