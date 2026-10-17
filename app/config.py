"""
Application configuration loaded from environment variables via Pydantic Settings.

All Azure service endpoints, credentials, and tunable parameters are centralised here.
Uses Pydantic v2 strict validation so missing or malformed config fails fast at startup
rather than silently at runtime.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Singleton configuration — loaded once at import time from env vars / .env file.

    Naming convention:
      FRR_ prefix = Financial Research RAG (project-specific)
      AZURE_ prefix = standard Azure environment variables
    """

    model_config = SettingsConfigDict(
        env_prefix="FRR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,  # Immutable after construction — prevents runtime config drift
    )

    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    # These map to your AOAI resource in the Azure portal.
    # We require separate deployment names because a single AOAI resource can host
    # multiple model deployments (gpt-4o, text-embedding-ada-002, etc.).
    azure_openai_endpoint: str = Field(
        default="",
        description=(
            "https://<your-resource>.openai.azure.com/ — empty means Azure OpenAI is not "
            "configured, and the pipeline degrades to keyword-only retrieval."
        ),
    )
    azure_openai_api_version: str = Field(
        default="2025-01-01-preview",
        description="Azure OpenAI API version — pin to a specific date for stability.",
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4o",
        description="Deployment name for the chat completion model (not the base model name).",
    )
    azure_openai_embedding_deployment_name: str = Field(
        default="text-embedding-ada-002",
        description="Deployment name for the embedding model.",
    )
    azure_openai_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retries for Azure OpenAI calls before falling back.",
    )

    # ── Azure AI Search ────────────────────────────────────────────────────────
    azure_search_endpoint: str = Field(
        default="",
        description=(
            "https://<your-service>.search.windows.net — empty means Azure AI Search is not "
            "configured."
        ),
    )
    azure_search_index_name: str = Field(
        default="financial-documents",
        description="Name of the search index — created automatically if it doesn't exist.",
    )
    azure_search_api_version: str = Field(
        default="2024-11-01-preview",
        description=(
            "Search API version. Preview is required for Semantic Ranker + vector features."
        ),
    )
    azure_search_semantic_config_name: str = Field(
        default="financial-semantic-config",
        description="Semantic ranker configuration name defined in the index schema.",
    )

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size_small: int = Field(
        default=512,
        ge=128,
        le=1024,
        description="Small chunk size in tokens — used for dense retrieval precision.",
    )
    chunk_size_large: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Large (parent) chunk size in tokens — provides LLM with full context window.",
    )
    chunk_overlap_tokens: int = Field(
        default=128,
        ge=0,
        le=512,
        description="Overlap between consecutive chunks to avoid boundary information loss.",
    )

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of candidate documents retrieved from hybrid search before reranking.",
    )
    retrieval_top_k_reranked: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents kept after Semantic Ranker reranking.",
    )
    retrieval_min_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum reranking score threshold — results below this are discarded.",
    )
    retrieval_vector_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for vector component in hybrid search (0 = BM25-only, 1 = vector-only)."
        ),
    )

    # ── Generation ─────────────────────────────────────────────────────────────
    generation_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature — low value keeps the model grounded in retrieved context.",
    )
    generation_max_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Maximum tokens in the generated response.",
    )
    generation_stream: bool = Field(
        default=True,
        description="Whether to stream responses via SSE by default.",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_port: int = Field(default=8000, ge=1024, le=65535)
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
    )
    app_cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins — tighten in production.",
    )

    # ── Azure Identity ─────────────────────────────────────────────────────────
    # For local dev, DefaultAzureCredential picks up Azure CLI auth, VS Code, or env vars.
    # For production, deploy a Managed Identity and assign RBAC roles to AOAI + Search.
    azure_credential_managed_identity_client_id: str | None = Field(
        default=None,
        description=(
            "Client ID of the User-Assigned Managed Identity "
            "for production. None = DefaultAzureCredential."
        ),
    )

    # ── Observability ──────────────────────────────────────────────────────────
    otel_service_name: str = Field(
        default="financial-research-rag",
        description="OpenTelemetry service name — appears in traces and metrics.",
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        description="OTLP endpoint for traces. None = stdout-only tracing.",
    )

    # ── Cross-field validation ─────────────────────────────────────────────────
    @field_validator("azure_openai_endpoint", "azure_search_endpoint")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure endpoints are HTTPS — Azure doesn't serve over plain HTTP."""
        if not v:
            # An unset endpoint is allowed. The application must be importable and
            # testable without Azure credentials, and the test suite runs fully
            # mocked. Each service validates that its own endpoint is configured on
            # first use, which is where a missing value can still fail loudly.
            return v
        if not v.startswith("https://"):
            msg = f"Azure endpoint must use HTTPS: {v}"
            raise ValueError(msg)
        return v.rstrip("/")

    @field_validator("chunk_size_large")
    @classmethod
    def large_chunk_must_exceed_small(cls, v: int, info: ValidationInfo) -> int:
        """Parent chunks must be larger than child chunks — otherwise the hierarchy collapses."""
        small = info.data.get("chunk_size_small", 512)
        if v <= small:
            msg = f"chunk_size_large ({v}) must exceed chunk_size_small ({small})"
            raise ValueError(msg)
        return v


# Module-level singleton — import { settings } anywhere for consistent config.
settings = Settings()
