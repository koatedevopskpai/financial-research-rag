from pathlib import Path

import pytest

from app.ingestion.chunker import split_into_chunks
from app.tokenizer_assets import get_tokenizer


def test_missing_tokenizer_asset_fails_without_implicit_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="--prepare"):
        get_tokenizer()


def test_unicode_chunk_windows_progress_and_stay_in_budget() -> None:
    tokenizer = get_tokenizer()
    text = "财务报告收入增长。" * 20
    spans = split_into_chunks(text, tokenizer, max_tokens=8, overlap_tokens=6, boundaries=[])
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(end > start for start, end in spans)
    assert all(spans[i + 1][0] > spans[i][0] for i in range(len(spans) - 1))
    assert all(len(tokenizer.encode(text[start:end])) <= 8 for start, end in spans)
    covered = {position for start, end in spans for position in range(start, end)}
    assert covered == set(range(len(text)))
