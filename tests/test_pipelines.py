"""
End-to-end pipeline tests with mocked Azure dependencies.

These tests verify the pipeline orchestration logic without requiring Azure credentials.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.documents import SourceMetadata, Chunk
from app.models.requests import QueryRequest, QueryResponse


class TestDocumentPipeline:
    """Tests for the document ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_process_document_with_empty_text_returns_error(self):
        """An empty or unparseable document should return success=False."""
        from app.pipelines.document_pipeline import DocumentPipeline

        pipeline = DocumentPipeline.__new__(DocumentPipeline)
        pipeline.index_manager = MagicMock()
        pipeline.embedder = MagicMock()
        pipeline.search_index = MagicMock()

        with patch.object(pipeline, "_parse_document", return_value=""):
            result = await pipeline.process_document(
                contents=b"",
                filename="empty.txt",
                file_type="txt",
            )

        assert result.success is False
        assert "empty text" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_process_document_happy_path(self, sample_chunks):
        """A valid document should be chunked, embedded, and indexed successfully."""
        from app.pipelines.document_pipeline import DocumentPipeline

        pipeline = DocumentPipeline.__new__(DocumentPipeline)
        pipeline.index_manager = MagicMock()
        pipeline.embedder = AsyncMock()
        pipeline.embedder.embed.return_value = sample_chunks
        pipeline.search_index = MagicMock()

        with (
            patch.object(pipeline, "_parse_document", return_value="Some text."),
            patch("app.pipelines.document_pipeline.chunk_document", return_value=sample_chunks),
        ):
            result = await pipeline.process_document(
                contents=b"Some text.",
                filename="test.pdf",
                file_type="pdf",
            )

        assert result.success is True
        assert result.total_chunks == len(sample_chunks)


class TestQueryPipeline:
    """Tests for the query pipeline."""

    @pytest.mark.asyncio
    async def test_query_returns_response(self, mock_search_client, mock_search_results):
        """A query should return a QueryResponse with sources."""
        from app.pipelines.query_pipeline import QueryPipeline

        pipeline = QueryPipeline.__new__(QueryPipeline)
        pipeline.search_client = mock_search_client
        pipeline.hybrid_search = MagicMock()
        pipeline.hybrid_search.search = AsyncMock(return_value=mock_search_results)
        pipeline.reranker = MagicMock()
        pipeline.reranker.rerank = AsyncMock(return_value=mock_search_results)
        pipeline.rag_chain = MagicMock()
        pipeline.rag_chain.generate = AsyncMock(
            return_value=QueryResponse(
                answer="Apple reported $395.6B in revenue.",
                sources=[],
                total_sources_retrieved=2,
            )
        )

        request = QueryRequest(query="What was Apple's revenue?")
        response = await pipeline.query(request)

        assert isinstance(response, QueryResponse)
        assert "revenue" in response.answer.lower()

    @pytest.mark.asyncio
    async def test_query_with_no_results_returns_empty_response(self, mock_search_client):
        """When no documents match, return an appropriate message."""
        from app.pipelines.query_pipeline import QueryPipeline

        pipeline = QueryPipeline.__new__(QueryPipeline)
        pipeline.search_client = mock_search_client
        pipeline.hybrid_search = MagicMock()
        pipeline.hybrid_search.search = AsyncMock(return_value=[])
        pipeline.reranker = MagicMock()
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        request = QueryRequest(query="Nothing matches this")
        response = await pipeline.query(request)

        assert "No relevant documents found" in response.answer
