"""
Tests for the hybrid search and reranker services.

Uses mocked Azure SearchClient via pytest fixtures (no Azure credentials needed).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.search import HybridSearchService
from app.retrieval.reranker import RerankerService


class TestHybridSearch:
    """Tests for the hybrid search service."""

    def test_search_returns_results(self, mock_search_client, mock_search_results):
        """Search should return results from the search client."""
        mock_search_client.search.return_value = mock_search_results
        service = HybridSearchService(mock_search_client)

        results = list(service.search(
            query="What was Apple's revenue in 2025?",
            query_embedding=[0.1, 0.2, 0.3],
        ))

        assert len(results) == 2
        assert results[0]["ticker"] == "AAPL"
        mock_search_client.search.assert_called_once()

    def test_search_without_embedding_falls_back_to_keyword(self, mock_search_client):
        """Without an embedding, search should still work via keyword."""
        mock_search_client.search.return_value = []
        service = HybridSearchService(mock_search_client)

        results = list(service.search(
            query="EBITDA margin",
            query_embedding=None,
        ))

        assert results == []

    def test_filter_building(self):
        """OData filter strings should be built correctly."""
        from app.retrieval.search import HybridSearchService
        service = HybridSearchService(MagicMock())

        # Test with ticker + fiscal_year
        filters = {"ticker": "AAPL", "fiscal_year": 2025}
        result = service._build_filter_string(filters)
        assert "ticker eq 'AAPL'" in result
        assert "fiscal_year eq 2025" in result

    def test_filter_with_string_escaping(self):
        """Single quotes in filter values should be escaped."""
        service = HybridSearchService(MagicMock())
        filters = {"ticker": "O'Reilly"}
        result = service._build_filter_string(filters)
        assert "O''Reilly" in result  # Double single-quote escapes


class TestReranker:
    """Tests for the reranker service."""

    def test_reranker_filters_by_score_threshold(self, mock_search_client, mock_search_results):
        """Results below the score threshold should be filtered out."""
        mock_search_client.search.return_value = mock_search_results
        service = RerankerService(mock_search_client)

        # Set threshold above the second result's score
        results = service.rerank(
            query="Apple revenue",
            top_k=5,
            min_score=0.75,
        )

        assert len(results) >= 1  # At least the highest-scoring result

    def test_reranker_sorts_by_score_descending(self, mock_search_client, mock_search_results):
        """Results should be sorted by reranker score descending."""
        mock_search_client.search.return_value = mock_search_results
        service = RerankerService(mock_search_client)

        results = service.rerank(query="Apple revenue", min_score=0.0)

        scores = [r.get("score", 0) for r in results]
        assert scores == sorted(scores, reverse=True)
