"""
Azure AI Search indexer — creates the index (if absent) and pushes documents.

Design decisions:
  1. Index creation is idempotent — we attempt to create on every startup,
     and if it already exists, we catch the 409 and proceed.
  2. Batch push — documents are uploaded in batches of 1000 to stay under
     Azure's per-batch limits and improve throughput.
  3. Separate fields for vector + non-vector — the index schema explicitly
     separates searchable text fields from vector fields so that hybrid
     search queries can target each appropriately.
  4. Semantic ranker config is set at index creation — this enables the
     Semantic Ranker reranking step in the retrieval pipeline.
"""

from __future__ import annotations

from typing import Any

from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import IndexingResult as AzureIndexingResult

from app.config import settings
from app.models.documents import Chunk
from app.utils.logging import get_logger

logger = get_logger(__name__)

# ── Index Schema ───────────────────────────────────────────────────────────────
# This schema is the contract between ingestion and retrieval.
# Every field maps 1:1 to a Pydantic model field in documents.py.
# Changing the schema requires a delete-recreate or an additive migration.

INDEX_FIELDS: list[SearchField] = [
    SimpleField(
        name="chunk_id",
        type="Edm.String",
        key=True,  # Primary key for AI Search
        filterable=True,
        sortable=True,
    ),
    SimpleField(
        name="parent_chunk_id",
        type="Edm.String",
        filterable=True,
    ),
    SimpleField(
        name="chunk_type",
        type="Edm.String",
        filterable=True,
    ),
    SearchableField(
        name="content",
        type="Edm.String",
        searchable=True,  # Full-text search on chunk content
        analyzer_name="en.microsoft",  # Microsoft's English analyzer — handles financial terms well
    ),
    SimpleField(
        name="token_count",
        type="Edm.Int32",
        filterable=True,
        sortable=True,
    ),
    SimpleField(
        name="chunk_index",
        type="Edm.Int32",
        filterable=True,
        sortable=True,
    ),
    SearchField(
        name="embedding",
        type="Collection(Edm.Single)",
        searchable=True,  # Vector search
        vector_search_dimensions=1536,  # Ada-002 outputs 1536-dim vectors
        vector_search_profile_name="financial-vector-profile",
    ),
    SimpleField(
        name="document_id",
        type="Edm.String",
        filterable=True,
    ),
    SimpleField(
        name="filename",
        type="Edm.String",
        filterable=True,
        sortable=True,
    ),
    SimpleField(
        name="ticker",
        type="Edm.String",
        filterable=True,
    ),
    SimpleField(
        name="fiscal_year",
        type="Edm.Int32",
        filterable=True,
        sortable=True,
    ),
    SimpleField(
        name="document_type",
        type="Edm.String",
        filterable=True,
    ),
]


class SearchIndexManager:
    """
    Manages the Azure AI Search index lifecycle: creation, document upload, and queries.

    This is a shared resource between the ingestion pipeline (writes) and the
    retrieval pipeline (reads). We expose a SearchClient for query operations
    and an indexer for upload operations.
    """

    def __init__(self) -> None:
        if not settings.azure_search_endpoint:
            raise ValueError("FRR_AZURE_SEARCH_ENDPOINT is required for indexing")
        self.credential = DefaultAzureCredential(
            managed_identity_client_id=settings.azure_credential_managed_identity_client_id
        )
        self.index_client = SearchIndexClient(
            endpoint=settings.azure_search_endpoint,
            credential=self.credential,
            api_version=settings.azure_search_api_version,
        )
        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        """
        Create the search index if it doesn't exist.

        This is intentionally idempotent — the index may already exist from a
        previous deployment. We attempt creation on every startup and handle the
        409 Conflict gracefully.
        """
        index_name = settings.azure_search_index_name

        try:
            self.index_client.get_index(index_name)
            logger.info("Search index already exists", index=index_name)
            return
        except (ResourceNotFoundError, ServiceRequestError):
            logger.info("Creating search index", index=index_name)

        index = SearchIndex(
            name=index_name,
            fields=INDEX_FIELDS,
            vector_search=VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="financial-hnsw-config",
                        parameters=HnswParameters(
                            m=16,
                            ef_construction=400,
                            metric="cosine",
                        ),
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="financial-vector-profile",
                        algorithm_configuration_name="financial-hnsw-config",
                    )
                ],
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name=settings.azure_search_semantic_config_name,
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="filename"),
                            content_fields=[SemanticField(field_name="content")],
                        ),
                    )
                ]
            ),
        )

        try:
            self.index_client.create_index(index)
            logger.info("Search index created successfully", index=index_name)
        except Exception as exc:
            # 409 Conflict means the index already exists — not an error at startup.
            if getattr(exc, "status_code", None) == 409:
                logger.info("Search index already exists (409)", index=index_name)
            else:
                logger.warning(
                    "Failed to create search index — will attempt operations anyway",
                    index=index_name,
                    error=str(exc),
                )

    def get_search_client(self) -> SearchClient:
        """Return a SearchClient for this index — used by the retrieval pipeline."""
        return SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=self.credential,
            api_version=settings.azure_search_api_version,
        )

    def _chunk_to_dict(self, chunk: Chunk) -> dict[str, Any]:
        """
        Convert a Pydantic Chunk to a dictionary for AI Search upload.

        We exclude None values (especially embedding=None for failed embeddings)
        because AI Search rejects null vector fields.
        """
        allowed = {field.name for field in INDEX_FIELDS}
        return {
            key: value
            for key, value in chunk.model_dump(exclude_none=True).items()
            if key in allowed
        }

    async def upload_documents(self, chunks: list[Chunk]) -> list[AzureIndexingResult]:
        """
        Upload chunks to Azure AI Search in batches.

        Args:
            chunks: List of Chunk objects to upload.

        Returns:
            List of Azure per-document IndexingResult objects.
        """
        search_client = self.get_search_client()
        documents = [self._chunk_to_dict(c) for c in chunks]

        # Azure AI Search batch limit is 1000 documents per request.
        batch_size = 1000
        total_results: list[AzureIndexingResult] = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                result = search_client.upload_documents(documents=batch)
                if len(result) != len(batch):
                    raise ValueError("Azure upload returned an incomplete result set")
                total_results.extend(result)
                logger.debug("Batch uploaded", batch_index=i // batch_size, count=len(batch))
            except Exception as exc:
                logger.error(
                    "Batch upload failed",
                    batch_index=i // batch_size,
                    error=str(exc),
                )
                # Re-raise — the pipeline handles this upstream
                raise

        success_count = sum(1 for r in total_results if r.succeeded)
        logger.info(
            "Document upload complete",
            total=len(documents),
            succeeded=success_count,
            failed=len(documents) - success_count,
        )
        return total_results
