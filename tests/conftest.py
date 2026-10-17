"""
Test fixtures — mock Azure clients for deterministic, fast tests.

No Azure credentials are needed — all services are mocked using respx (HTTP mocks).
"""

from __future__ import annotations

import socket
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from app.models.documents import Chunk, SourceMetadata
from app.tokenizer_assets import cache_path

cache_path()  # Explicit prepared cache for all tokenizer calls, including direct SDK tests.


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No implicit cloud/tokenizer calls in tests; local event-loop sockets are allowed."""
    connect = socket.socket.connect
    resolve = socket.getaddrinfo

    def guarded_connect(sock: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("external network is disabled during tests")
        connect(sock, address)

    def guarded_resolve(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None and host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("external DNS is disabled during tests")
        return resolve(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_resolve)


@pytest.fixture
def sample_source_metadata() -> SourceMetadata:
    """A realistic source metadata for testing."""
    return SourceMetadata(
        document_id="doc-001",
        filename="annual_report_2025.pdf",
        file_type="pdf",
        ticker="AAPL",
        fiscal_year=2025,
        document_type="annual_report",
    )


@pytest.fixture
def sample_chunks(sample_source_metadata: SourceMetadata) -> list[Chunk]:
    """A set of realistic chunks for testing the pipeline."""
    meta = sample_source_metadata
    chunks = [
        Chunk(
            chunk_id="parent-001",
            chunk_type="parent",
            content=(
                "Apple Inc. reported strong financial results for fiscal year 2025. "
                "Total net sales were $395.6 billion, representing a 5% increase year-over-year. "
                "The company generated $108.9 billion in operating cash flow and returned "
                "over $25 billion to shareholders through dividends and share repurchases. "
                "Services revenue reached an all-time high of $96.7 billion."
            ),
            token_count=80,
            chunk_index=0,
            document_id=meta.document_id,
            filename=meta.filename,
            ticker=meta.ticker,
            fiscal_year=meta.fiscal_year,
            document_type=meta.document_type,
        ),
        Chunk(
            chunk_id="child-001",
            parent_chunk_id="parent-001",
            chunk_type="child",
            content=(
                "Apple Inc. reported strong financial results for fiscal year 2025. "
                "Total net sales were $395.6 billion."
            ),
            token_count=25,
            chunk_index=1,
            document_id=meta.document_id,
            filename=meta.filename,
            ticker=meta.ticker,
            fiscal_year=meta.fiscal_year,
            document_type=meta.document_type,
            embedding=[0.1] * 8,
            embedding_dimensions=8,
        ),
        Chunk(
            chunk_id="child-002",
            parent_chunk_id="parent-001",
            chunk_type="child",
            content=(
                "Services revenue reached an all-time high of $96.7 billion, "
                "driven by strong growth in the App Store and Apple Music."
            ),
            token_count=22,
            chunk_index=2,
            document_id=meta.document_id,
            filename=meta.filename,
            ticker=meta.ticker,
            fiscal_year=meta.fiscal_year,
            document_type=meta.document_type,
            embedding=[0.2] * 8,
            embedding_dimensions=8,
        ),
    ]
    return chunks


@pytest.fixture
def mock_search_results() -> list[dict[str, Any]]:
    """Mock Azure AI Search results for testing the retrieval pipeline."""
    return [
        {
            "chunk_id": "child-001",
            "parent_chunk_id": "parent-001",
            "chunk_type": "child",
            "content": (
                "Apple Inc. reported strong financial results for fiscal "
                "year 2025. Total net sales were $395.6 billion."
            ),
            "document_id": "doc-001",
            "filename": "annual_report_2025.pdf",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "document_type": "annual_report",
            "token_count": 25,
            "score": 0.89,
            "@search.reranker_score": 0.87,
        },
        {
            "chunk_id": "child-002",
            "parent_chunk_id": "parent-001",
            "chunk_type": "child",
            "content": "Services revenue reached an all-time high of $96.7 billion.",
            "document_id": "doc-001",
            "filename": "annual_report_2025.pdf",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "document_type": "annual_report",
            "token_count": 22,
            "score": 0.76,
            "@search.reranker_score": 0.72,
        },
    ]


@pytest_asyncio.fixture
async def mock_search_client() -> MagicMock:
    """Mock Azure SearchClient — returns controlled results for testing."""
    client = MagicMock()
    # Mock search returns
    client.search.return_value = []
    return client
