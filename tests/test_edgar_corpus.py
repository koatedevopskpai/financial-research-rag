"""Offline corpus contracts use self-authored HTML, never downloaded filing text."""

import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest
import respx

from app import edgar_corpus as corpus

HTML = b"<!DOCTYPE html><html><body>Self-authored revenue fixture.</body></html>"
URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"


def fixture_filing() -> corpus.Filing:
    return corpus.Filing(
        "AAPL",
        320193,
        "10-K",
        "2025-09-27",
        "2025-10-31",
        "0000320193-25-000079",
        URL,
        hashlib.sha256(HTML).hexdigest(),
        len(HTML),
    )


def test_reproduce_exact_bytes_and_detect_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    filing = fixture_filing()
    directory = tmp_path / "artifacts/corpus"
    corpus.reproduce([filing], directory, lambda _url: HTML)
    assert (directory / filing.filename).read_bytes() == HTML
    with pytest.raises(ValueError, match="changed or corrupted"):
        corpus.reproduce([filing], directory, lambda _url: HTML.replace(b"revenue", b"fiction"))
    assert (directory / filing.filename).read_bytes() == HTML


@pytest.mark.parametrize(
    "url",
    [
        "http://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl.htm",
        URL + "?other=true",
        URL.replace("www.sec.gov", "evil.test"),
        URL.replace("aapl-20250927.htm", "../outside.htm"),
        URL.replace("aapl-20250927.htm", "%2e%2e%2foutside.htm"),
    ],
)
def test_reject_unsafe_url(url: str) -> None:
    with pytest.raises(ValueError):
        corpus.validate_url(url)


def test_storage_must_stay_inside_ignored_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="ignored artifacts"):
        corpus.store_filing(fixture_filing(), HTML, tmp_path / "corpus")
    assert not (tmp_path / "corpus").exists()


def test_manifest_validates_identity_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    filing = fixture_filing()
    path.write_text(json.dumps({"schema_version": 1, "filings": [asdict(filing)]}))
    assert corpus.load_manifest(path) == [filing]
    for bad in [replace(filing, sha256="unknown"), replace(filing, cik=789019)]:
        path.write_text(json.dumps({"schema_version": 1, "filings": [asdict(bad)]}))
        with pytest.raises(ValueError):
            corpus.load_manifest(path)


@pytest.mark.parametrize("status", [302, 403, 429])
def test_http_failures_stop_without_retry(status: int) -> None:
    client = corpus.SECClient("Portfolio research contact@example.test")
    with respx.mock() as mock:
        route = mock.get(URL).mock(return_value=httpx.Response(status))
        with pytest.raises(httpx.HTTPStatusError):
            client.fetch(URL)
        assert route.call_count == 1
        assert client.requests == 1


def test_bounded_response_and_request_count(monkeypatch: pytest.MonkeyPatch) -> None:
    client = corpus.SECClient("Portfolio research contact@example.test")
    monkeypatch.setattr(corpus, "MAX_BYTES", 10)
    with respx.mock() as mock:
        route = mock.get(URL).mock(return_value=httpx.Response(200, content=HTML))
        with pytest.raises(ValueError, match="byte limit"):
            client.fetch(URL)
        client.requests = corpus.MAX_REQUESTS
        with pytest.raises(RuntimeError, match="cap"):
            client.fetch(URL)
        assert route.call_count == 1


@pytest.mark.parametrize(
    "data", [b"<html>partial", b"denied", b"<html>Undeclared Automated Tool</html>"]
)
def test_invalid_html_is_not_a_filing(data: bytes) -> None:
    with pytest.raises(ValueError):
        corpus.validate_html(data)


def test_missing_file_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    filing = fixture_filing()
    directory = corpus.ignored_store(tmp_path / "artifacts/missing")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "filings": [asdict(filing)]}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["edgar_corpus", "verify", "--manifest", str(manifest), "--directory", str(directory)],
    )
    with pytest.raises(FileNotFoundError):
        corpus.main()


def test_failed_download_never_publishes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    manifest = tmp_path / "manifest.json"
    client = corpus.SECClient("Portfolio research contact@example.test")
    with respx.mock() as mock:
        mock.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
            return_value=httpx.Response(403)
        )
        with pytest.raises(httpx.HTTPStatusError):
            corpus.create_manifest(client, tmp_path / "artifacts/download", manifest)
    assert not manifest.exists()


def test_serial_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = corpus.SECClient("Portfolio research contact@example.test")
    client.last_request = 5.0
    sleeps: list[float] = []
    monkeypatch.setattr(time, "monotonic", lambda: 5.25)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    with respx.mock() as mock:
        mock.get(URL).mock(return_value=httpx.Response(200, content=HTML))
        assert client.fetch(URL) == HTML
    assert sleeps == [0.75]
