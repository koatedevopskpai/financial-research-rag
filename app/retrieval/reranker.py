"""
Semantic Ranker reranking — applies Azure AI Search's Semantic Ranker to improve result ordering.

Design decision — Semantic Ranker over custom cross-encoder:
  Azure's Semantic Ranker is a managed cross-encoder model fine-tuned for search.
  Using it avoids:
    - Hosting our own cross-encoder model (GPU cost, latency)
    - Training/fine-tuning for the financial domain (Azure's model generalises well)
    - Maintaining model versions

  The trade-off: we pay per query for semantic ranking. For investment research volume
  (hundreds, not millions of queries/day), this is negligible.

Processing flow:
  1. Hybrid search returns top-K candidates (e.g. 20)
  2. Semantic Ranker re-scores them using cross-attention between query and document
  3. We apply a minimum score threshold to discard low-relevance results
  4. The top N highest-scoring results are returned to the LLM for generation

Score threshold rationale:
  Financial hallucinations are costly. By discarding low-confidence results,
  we reduce the chance that the LLM invents answers based on tangentially related text.
"""

from __future__ import annotations

from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RerankerService:
    """
    Applies Azure AI Search Semantic Ranker to rerank hybrid search results.

    The reranker operates at the query level — it re-executes the query with
    query_type=Semantic and uses the Semantic Configuration defined on the index.
    """

    def __init__(self, search_client: SearchClient):
        self.search_client = search_client
        self.semantic_config = settings.azure_search_semantic_config_name
        self.top_k = settings.retrieval_top_k_reranked
        self.min_score = settings.retrieval_min_score

    async def rerank(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict]:
        """
        Execute a semantic reranking pass over the index.

        This is a standalone rerank call — used when you want the Semantic Ranker
        to re-score the hybrid search results. The ranker re-ranks the top-k results
        from the initial hybrid query.

        Args:
            query: The original user query.
            top_k: Number of reranked results to return.
            min_score: Minimum reranker score threshold.

        Returns:
            Reranked list of results sorted by @search.reranker_score descending.
        """
        k = top_k or self.top_k
        threshold = min_score or self.min_score

        results = self.search_client.search(
            search_text=query,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name=self.semantic_config,
            top=k,
            select=[
                "chunk_id",
                "parent_chunk_id",
                "chunk_type",
                "content",
                "document_id",
                "filename",
                "ticker",
                "fiscal_year",
                "token_count",
            ],
        )

        # Filter by reranker score threshold
        filtered: list[dict] = []
        for r in results:
            # Semantic Ranker returns scores in @search.reranker_score
            score = r.get("@search.reranker_score", 0.0)
            if score >= threshold:
                r["score"] = score  # Normalise the score field
                filtered.append(r)

        # Sort by score descending (should already be ordered, but be explicit)
        filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        logger.debug(
            "Reranking complete",
            candidates=results.get_count() if hasattr(results, "get_count") else 0,
            passed_threshold=len(filtered),
            threshold=threshold,
        )

        return filtered
