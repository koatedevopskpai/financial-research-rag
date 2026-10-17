"""
RAG generation chain — retrieve → augment → generate with streaming support.

Design decisions:
  1. Streaming-first — the generation endpoint streams tokens via SSE so that
     users see partial results immediately. Financial analysts are impatient with
     loading spinners.
  2. Degraded mode detection — if Azure OpenAI is unavailable, the chain returns
     an extractive summary (keyword-matching) instead of a generative answer.
  3. Context is assembled from supplied results. A 70/20/10 context-budget policy
     is not implemented and is not claimed.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncAzureOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.generation.prompts import CONTEXT_TEMPLATE, SYSTEM_PROMPT, build_context_from_results
from app.models.requests import ChunkResult, QueryResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RAGChain:
    """
    The RAG generation step — takes retrieved context + user query → LLM response.

    This is the final stage of the RAG pipeline. It combines the retrieved chunks
    with the user's query and sends them to the LLM for answer generation.
    """

    def __init__(self) -> None:
        if not settings.azure_openai_endpoint:
            raise ValueError("FRR_AZURE_OPENAI_ENDPOINT is required for generation")
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        credential = DefaultAzureCredential(
            managed_identity_client_id=settings.azure_credential_managed_identity_client_id
        )

        self.client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_ad_token_provider=get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            ),
        )
        self.deployment_name = settings.azure_openai_deployment_name
        self.temperature = settings.generation_temperature
        self.max_tokens = settings.generation_max_tokens

    async def generate(self, query: str, results: list[dict[str, Any]]) -> QueryResponse:
        """
        Generate a complete (non-streaming) response.

        Args:
            query: The user's original question.
            results: Reranked search results from the retrieval pipeline.

        Returns:
            QueryResponse with answer and source citations.
        """
        start = time.perf_counter()
        degraded = False
        context = build_context_from_results(results)
        prompt = CONTEXT_TEMPLATE.format(context=context, query=query)

        try:
            response = await self._call_llm(prompt)
            answer = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("LLM generation failed, returning degraded response", error=str(exc))
            answer = self._degraded_response(query, results)
            degraded = True

        sources = self._build_sources(results)
        return QueryResponse(
            answer=answer,
            sources=sources,
            total_sources_retrieved=len(results),
            degraded_mode=degraded,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
        )

    async def stream_generate(
        self, query: str, results: list[dict[str, Any]]
    ) -> AsyncIterator[dict[str, str]]:
        """
        Generate a streaming response, yielding tokens as they arrive.

        Each yield is a dict with structure:
          {"event": "token", "data": {"token": "the next word"}}
        Final yield:
          {"event": "done", "data": {"sources": [...], "total_sources_retrieved": N}}
        """
        context = build_context_from_results(results)
        prompt = CONTEXT_TEMPLATE.format(context=context, query=query)

        try:
            stream = await self._call_llm_stream(prompt)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield {
                        "event": "token",
                        "data": json.dumps({"token": chunk.choices[0].delta.content}),
                    }
        except Exception as exc:
            logger.warning("Streaming generation failed", error=str(exc))
            yield {
                "event": "token",
                "data": json.dumps({"token": self._degraded_response(query, results)}),
            }

        # Final metadata frame
        sources = self._build_sources(results)
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "sources": [s.model_dump() for s in sources],
                    "total_sources_retrieved": len(results),
                }
            ),
        }

    @retry(
        stop=stop_after_attempt(settings.azure_openai_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError)
        ),
        reraise=True,
    )
    async def _call_llm(self, prompt: str) -> ChatCompletion:
        """Non-streaming LLM call with retry."""
        return await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @retry(
        stop=stop_after_attempt(settings.azure_openai_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError)
        ),
        reraise=True,
    )
    async def _call_llm_stream(self, prompt: str) -> AsyncStream[ChatCompletionChunk]:
        """Streaming LLM call with retry."""
        return await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

    def _degraded_response(self, query: str, results: list[dict[str, Any]]) -> str:
        """
        Fallback response when Azure OpenAI is unavailable.

        Instead of failing, we return an extractive summary — the most relevant
        passages from the search results. This maintains utility even when the
        LLM backend is down.
        """
        if not results:
            return (
                "Failed to generate an AI-powered response (Azure OpenAI unavailable). "
                "No relevant documents found in keyword fallback search either."
            )

        passages = []
        for r in results[:3]:  # Top 3 passages
            content = r.get("content", "")
            filename = r.get("filename", "unknown")
            passages.append(f"[From {filename}]: {content[:500]}")

        return (
            f"Azure OpenAI is currently unavailable. "
            f"Below are the most relevant passages retrieved from the document index "
            f"for your query: '{query}'\n\n"
            + "\n\n".join(passages)
            + (
                "\n\n[Note: This is a degraded-mode response. Please "
                "retry when the AI service is available.]"
            )
        )

    def _build_sources(self, results: list[dict[str, Any]]) -> list[ChunkResult]:
        """Convert raw search results to ChunkResult models for the response."""
        sources: list[ChunkResult] = []
        for r in results:
            sources.append(
                ChunkResult(
                    chunk_id=r.get("chunk_id", ""),
                    document_id=r.get("document_id", ""),
                    content=r.get("content", "")[:300],  # Truncate for display
                    score=r.get("score", 0.0),
                    filename=r.get("filename", "unknown"),
                    ticker=r.get("ticker"),
                )
            )
        return sources
