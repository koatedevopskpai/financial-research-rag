"""
Azure OpenAI embedding service with batch processing, retry, and circuit breaker.

Design decisions:
  1. Batch processing — Azure OpenAI embedding API supports up to 16 inputs per call.
     We batch aggressively to minimise HTTP round-trips while staying under limits.
  2. Deterministic retry with exponential backoff — transient failures (429, 5xx)
     are retried up to N times before falling back.
  3. Circuit breaker — after N consecutive failures, the breaker opens for 30s
     to prevent cascading failures. During the open window, embeddings return None
     and the caller degrades gracefully (vector search falls back to keyword).
  4. No fallback embedding model — if AOAI is down, we don't attempt a local model.
     The system returns "degraded mode" responses. This is intentional: financial
     research demands consistency, and swapping to a different embedding model
     mid-pipeline would change the vector space and produce unreliable results.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

import httpx
from openai import AsyncAzureOpenAI
from openai.types import CreateEmbeddingResponse
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.models.documents import Chunk
from app.utils.logging import get_logger

logger = get_logger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


class CircuitBreaker:
    """
    Simple circuit breaker for Azure OpenAI calls.

    States:
      CLOSED  — normal operation, requests pass through
      OPEN    — failures exceed threshold, requests are short-circuited
      HALF_OPEN — after timeout, one test request is allowed through

    This prevents the "thundering herd" problem: if AOAI is down, hundreds of
    concurrent retries would waste connections and prolong recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN

    async def call(self, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute func respecting circuit breaker state."""
        if self.state == "OPEN":
            if time.monotonic() - (self.last_failure_time or 0) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker: HALF_OPEN — allowing test request")
            else:
                remaining = self.recovery_timeout - (
                    time.monotonic() - (self.last_failure_time or 0)
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker OPEN — {self.failure_count} failures, "
                    f"retry in {remaining:.0f}s"
                )

        try:
            result = await func(*args, **kwargs)
            # Success — reset breaker
            self.failure_count = 0
            self.state = "CLOSED"
            return result
        except Exception:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(
                    "Circuit breaker: OPEN — threshold reached",
                    failures=self.failure_count,
                )
            raise


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is in OPEN state and rejects a call."""


class EmbeddingService:
    """
    Batched embedding service for Azure OpenAI.

    Usage:
        embedder = EmbeddingService()
        chunks = embedder.embed(chunks)  # Modifies chunks in-place
    """

    def __init__(self) -> None:
        if not settings.azure_openai_endpoint:
            raise ValueError("FRR_AZURE_OPENAI_ENDPOINT is required for embedding")
        # Use Azure AD auth (DefaultAzureCredential) by default.
        # Falls through from Managed Identity → Azure CLI → environment variables.
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
        self.embedding_model = settings.azure_openai_embedding_deployment_name
        self.max_batch_size = 16  # AOAI limit for embedding API
        self.circuit_breaker = CircuitBreaker()

    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Generate embeddings for all child chunks.

        Parent chunks do NOT receive embeddings — they are not used for vector search.
        Child chunks are the retrieval granularity; parents are for context only.
        """
        to_embed = [c for c in chunks if c.chunk_type == "child"]
        if not to_embed:
            return chunks

        logger.info("Generating embeddings", count=len(to_embed))

        # Process in batches
        for i in range(0, len(to_embed), self.max_batch_size):
            batch = to_embed[i : i + self.max_batch_size]
            texts = [c.content for c in batch]

            try:
                embeddings = await self._embed_batch(texts)
                for chunk, embedding in zip(batch, embeddings, strict=True):
                    chunk.embedding = embedding
                    chunk.embedding_dimensions = len(embedding)
            except (CircuitBreakerOpenError, httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                logger.warning(
                    "Embedding batch failed — chunks will use keyword-only search",
                    batch_index=i,
                    error=str(exc),
                )
                # Chunks without embeddings fall back to BM25-only retrieval.
                # They don't fail — they're just less precise.
                for chunk in batch:
                    chunk.embedding = None

        embedded_count = sum(1 for c in to_embed if c.embedding is not None)
        failed_count = len(to_embed) - embedded_count
        logger.info("Embedding complete", embedded=embedded_count, failed=failed_count)

        return chunks

    @retry(
        stop=stop_after_attempt(settings.azure_openai_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError)
        ),
        reraise=True,
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Execute a single embedding API call with circuit breaker protection.

        The retry decorator handles transient failures (HTTP 429, 5xx).
        The circuit breaker handles sustained failures (repeated 429s, network down).
        """

        async def _do_request() -> CreateEmbeddingResponse:
            return await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )

        response = await self.circuit_breaker.call(_do_request)
        return [d.embedding for d in response.data]
