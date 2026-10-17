"""
Hybrid search pipeline — combines vector search (semantic) with BM25 full-text search.

Design decision — hybrid with weighted scoring:
  - Vector search captures semantic similarity ("companies with growing margins")
  - Full-text search captures exact term matching ("EBITDA", "ticker: AAPL")
  - The weighting (configurable via retrieval_vector_weight) balances the two

Financial domain note:
  Financial queries are often term-heavy ("What was the Q3 2025 EBITDA margin for AAPL?").
  Pure vector search can miss exact matches on tickers or fiscal periods. Hybrid search
  ensures that documents containing exact matches are surfaced regardless of semantic proximity.

Results are returned unranked — the Semantic Ranker (reranker.py) handles final ordering.
"""

from __future__ import annotations

from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryType,
    VectorQuery,
    VectorizedQuery,
)

from app.config import settings


class HybridSearchService:
    """
    Executes hybrid search queries against Azure AI Search.

    The service handles:
      1. Vector query construction (from user query embedding)
      2. Full-text query construction (raw user query)
      3. Combining both via Azure's HybridSearch API
      4. Optional OData filter application for time/company scoping
    """

    def __init__(self, search_client: SearchClient):
        self.search_client = search_client
        self.semantic_config = settings.azure_search_semantic_config_name
        self.top_k = settings.retrieval_top_k
        self.vector_weight = settings.retrieval_vector_weight

    async def search(
        self,
        query: str,
        query_embedding: list[float] | None,
        top_k: int | None = None,
        filters: dict[str, str | int | None] | None = None,
    ) -> list[dict]:
        """
        Execute a hybrid search against Azure AI Search.

        Args:
            query: The raw text query from the user.
            query_embedding: Vector embedding of the query (None = keyword-only search).
            top_k: Number of results to return (defaults to settings.retrieval_top_k).
            filters: Optional OData filter conditions.

        Returns:
            List of search result dictionaries, each containing the document fields + @search.score.
        """
        k = top_k or self.top_k

        # Build the vector query component
        vector_queries: list[VectorQuery] = []
        if query_embedding:
            vector_queries.append(
                VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=k,
                    fields="embedding",
                    weight=self.vector_weight,
                )
            )

        # Build OData filter string from structured filters
        filter_str = self._build_filter_string(filters) if filters else None

        # Execute hybrid search
        results = self.search_client.search(
            search_text=query,  # Full-text component
            vector_queries=vector_queries,
            query_type=QueryType.SIMPLE,
            top=k,
            filter=filter_str,
            include_total_count=True,
            select=[
                "chunk_id",
                "parent_chunk_id",
                "chunk_type",
                "content",
                "document_id",
                "filename",
                "ticker",
                "fiscal_year",
                "document_type",
                "token_count",
            ],
        )

        return list(results)

    def _build_filter_string(self, filters: dict[str, str | int | None]) -> str:
        """
        Convert structured filters to OData filter syntax.

        Example:
            {'ticker': 'AAPL', 'fiscal_year': 2025}
            -> "ticker eq 'AAPL' and fiscal_year eq 2025"
        """
        clauses: list[str] = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, str):
                # Escape single quotes in strings
                escaped = value.replace("'", "''")
                clauses.append(f"{key} eq '{escaped}'")
            elif isinstance(value, int):
                clauses.append(f"{key} eq {value}")
            # Add other types as needed
        return " and ".join(clauses) if clauses else ""
