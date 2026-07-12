"""
End-to-end query pipeline: embed → search → rerank → generate.

This pipeline is the entry point for all user queries.
Like the document pipeline, each step is independently replaceable.

Processing flow:
  1. Embed the user's query using the same embedding model as documents
  2. Execute hybrid search (vector + full-text) against Azure AI Search
  3. Rerank results using Azure Semantic Ranker
  4. Generate a response using the RAG chain (streaming or non-streaming)

Graceful degradation:
  - No embedding: fall back to keyword-only search (BM25)
  - No reranker: fall back to hybrid search scores
  - No LLM: return extractive summary from top passages
  - No search index: return error with degraded_mode=True
"""

from __future__ import annotations

from typing import AsyncIterator

from azure.search.documents import SearchClient

from app.generation.rag_chain import RAGChain
from app.models.requests import QueryRequest, QueryResponse, ChunkResult
from app.retrieval.search import HybridSearchService
from app.retrieval.reranker import RerankerService
from app.utils.logging import get_logger

logger = get_logger(__name__)


class QueryPipeline:
    """
    Orchestrates the full query pipeline for a single user question.
    """

    def __init__(self, search_client: SearchClient):
        self.search_client = search_client
        self.hybrid_search = HybridSearchService(search_client)
        self.reranker = RerankerService(search_client)
        self.rag_chain = RAGChain()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a single query and return a complete (non-streaming) response.

        Args:
            request: The user's query with parameters.

        Returns:
            QueryResponse with answer and source citations.
        """
        # Step 1: Embed query (if embedding service available)
        query_embedding = await self._embed_query(request.query)

        # Step 2: Hybrid search
        results = await self.hybrid_search.search(
            query=request.query,
            query_embedding=query_embedding,
            top_k=request.top_k,
            filters=request.filters,
        )
        total_retrieved = len(results)

        # Step 3: Rerank
        reranked = await self.reranker.rerank(
            query=request.query,
            top_k=request.top_k,
        )

        if not reranked:
            return QueryResponse(
                answer="No relevant documents found for your query.",
                sources=[],
                total_sources_retrieved=total_retrieved,
                processing_time_ms=0,
            )

        # Step 4: Generate
        response = await self.rag_chain.generate(
            query=request.query,
            results=reranked,
        )

        if not request.include_sources:
            response.sources = []

        response.total_sources_retrieved = total_retrieved
        return response

    async def stream_query(self, request: QueryRequest) -> AsyncIterator[dict]:
        """
        Process a single query and stream the response token-by-token via SSE.

        Yields dicts with event/data structure for sse-starlette.
        """
        # Steps 1-3 are the same as non-streaming
        query_embedding = await self._embed_query(request.query)
        results = await self.hybrid_search.search(
            query=request.query,
            query_embedding=query_embedding,
            top_k=request.top_k * 2,  # Retrieve more for reranking
            filters=request.filters,
        )
        reranked = await self.reranker.rerank(
            query=request.query,
            top_k=request.top_k,
        )

        if not reranked:
            yield {
                "event": "done",
                "data": '{"answer": "No relevant documents found.", "sources": [], "total_sources_retrieved": 0}',
            }
            return

        # Step 4: Stream generation
        async for event in self.rag_chain.stream_generate(
            query=request.query,
            results=reranked,
        ):
            yield event

    async def _embed_query(self, query: str) -> list[float] | None:
        """
        Embed the user query using Azure OpenAI.

        Returns None if embedding is unavailable (degraded mode).
        """
        try:
            from openai import AsyncAzureOpenAI
            from azure.identity import DefaultAzureCredential
            from app.config import settings

            credential = DefaultAzureCredential(
                managed_identity_client_id=settings.azure_credential_managed_identity_client_id
            )
            client = AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_ad_token_provider=credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                ),
            )
            response = await client.embeddings.create(
                model=settings.azure_openai_embedding_deployment_name,
                input=[query],
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("Query embedding failed — falling back to keyword search", error=str(exc))
            return None
