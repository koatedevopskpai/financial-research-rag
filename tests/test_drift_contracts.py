from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app import main
from app.generation.rag_chain import RAGChain
from app.ingestion.embedder import CircuitBreaker, CircuitBreakerOpenError
from app.ingestion.indexer import SearchIndexManager
from app.models.documents import Chunk, IndexingResult
from app.pipelines.document_pipeline import DocumentPipeline


async def test_breaker_observes_awaited_failure_and_short_circuits() -> None:
    breaker = CircuitBreaker(failure_threshold=1)
    failed = AsyncMock(side_effect=ValueError("fixture"))
    with pytest.raises(ValueError):
        await breaker.call(failed)
    assert breaker.state == "OPEN"
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(failed)
    assert failed.await_count == 1
    breaker.last_failure_time = (breaker.last_failure_time or 0) - 31
    success = AsyncMock(return_value="recovered")
    assert await breaker.call(success) == "recovered"
    assert breaker.state == "CLOSED"


async def test_generation_failure_sets_degraded_flag_and_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = RAGChain.__new__(RAGChain)
    monkeypatch.setattr(chain, "_call_llm", AsyncMock(side_effect=ValueError("fixture")))
    result = await chain.generate(
        "What is revenue?",
        [
            {
                "chunk_id": "fixture",
                "document_id": "fixture",
                "content": "Revenue was 10.",
                "filename": "fixture.txt",
                "score": 1.0,
            }
        ],
    )
    assert result.degraded_mode is True
    assert result.processing_time_ms >= 0
    assert "Revenue was 10." in result.answer
    assert result.sources[0].chunk_id == "fixture"


def test_index_documents_do_not_fabricate_zero_vectors(sample_chunks: list[Chunk]) -> None:
    manager = SearchIndexManager.__new__(SearchIndexManager)
    for chunk in sample_chunks:
        missing_vector = chunk.model_copy(update={"embedding": None, "embedding_dimensions": None})
        document = manager._chunk_to_dict(missing_vector)
        assert "embedding" not in document
        assert "embedding_dimensions" not in document
        assert document["chunk_id"] == chunk.chunk_id
        if chunk.embedding is not None:
            assert manager._chunk_to_dict(chunk)["embedding"] == chunk.embedding


def test_ingest_response_matches_declared_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = MagicMock()
    pipeline.process_document = AsyncMock(
        return_value=IndexingResult(
            document_id="fixture",
            filename="fixture.txt",
            parent_chunks_count=1,
            child_chunks_count=2,
            total_chunks=3,
            processing_time_ms=7,
            success=True,
        )
    )
    monkeypatch.setattr(main, "document_pipeline", pipeline)
    client = TestClient(main.app)
    response = client.post(
        "/ingest", files={"file": ("fixture.txt", b"Revenue was 10.", "text/plain")}
    )
    assert response.status_code == 200
    assert response.json()["total_chunks_indexed"] == 3
    assert response.json()["processing_time_ms"] == 7
    assert response.json()["results"][0]["success"] is True
    pipeline.process_document.assert_awaited_once()


def test_binary_parsers_fail_instead_of_indexing_placeholder() -> None:
    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    for file_type in ["pdf", "docx", "xlsx"]:
        with pytest.raises(ValueError, match="unsupported"):
            pipeline._parse_document(b"not parsed", file_type)
