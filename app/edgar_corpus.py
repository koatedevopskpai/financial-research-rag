"""Small, rate-limited EDGAR HTML corpus; filing bytes stay under ignored artifacts/."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 12_000_000
MAX_REQUESTS = 8
COMPANIES = {"AAPL": 320193, "MSFT": 789019}


def validate_url(url: str, *, metadata: bool = False) -> None:
    """Allow only canonical SEC archive HTML or the two submissions endpoints."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.query or parts.fragment or parts.username:
        raise ValueError("Only canonical HTTPS SEC URLs are allowed")
    if metadata:
        valid = parts.netloc == "data.sec.gov" and parts.path in {
            f"/submissions/CIK{cik:010d}.json" for cik in COMPANIES.values()
        }
    else:
        valid = parts.netloc == "www.sec.gov" and bool(
            re.fullmatch(
                r"/Archives/edgar/data/[0-9]+/[0-9]{18}/[A-Za-z0-9_.-]+\.html?", parts.path
            )
        )
    if not valid:
        raise ValueError("URL is outside the supported SEC corpus scope")


class SECClient:
    def __init__(self, user_agent: str) -> None:
        if (
            "@" not in user_agent
            or len(user_agent) < 20
            or "\n" in user_agent
            or "\r" in user_agent
        ):
            raise ValueError("Provide an organisation/project and contact email in --user-agent")
        self.user_agent = user_agent
        self.requests = 0
        self.last_request = 0.0

    def fetch(self, url: str, *, metadata: bool = False) -> bytes:
        validate_url(url, metadata=metadata)
        if self.requests >= MAX_REQUESTS:
            raise RuntimeError("Eight-request session cap reached")
        time.sleep(max(0.0, 1.0 - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        self.requests += 1
        # No retries or alternate hosts: 403/429 propagate and stop this invocation.
        chunks = []
        size = 0
        with httpx.stream(
            "GET",
            url,
            headers={"User-Agent": self.user_agent},
            timeout=30,
            follow_redirects=False,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("Corpus byte limit exceeded")
                chunks.append(chunk)
        data = b"".join(chunks)
        if not data or len(data) > MAX_BYTES:
            raise ValueError("Empty response or corpus byte limit exceeded")
        return data


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: int
    form: str
    period: str
    filing_date: str
    accession: str
    url: str
    sha256: str
    bytes: int

    def validate(self) -> None:
        validate_url(self.url)
        if self.ticker not in COMPANIES or self.cik != COMPANIES[self.ticker]:
            raise ValueError("Unknown company")
        if self.form != "10-K" or not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", self.accession):
            raise ValueError("Invalid annual filing identity")
        for date in (self.period, self.filing_date):
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date):
                raise ValueError("Invalid filing date")
        if not re.fullmatch(r"[a-f0-9]{64}", self.sha256) or not 0 < self.bytes <= MAX_BYTES:
            raise ValueError("Invalid content hash or byte count")
        expected = f"/Archives/edgar/data/{self.cik}/{self.accession.replace('-', '')}/"
        if not urlsplit(self.url).path.startswith(expected):
            raise ValueError("URL and filing identity disagree")

    @property
    def filename(self) -> str:
        self.validate()
        return f"{self.ticker}-{self.period}-{self.accession}.html"


def ignored_store(directory: Path) -> Path:
    directory = directory.resolve()
    artifacts = ROOT.resolve() / "artifacts"
    if directory == artifacts or artifacts not in directory.parents:
        raise ValueError("Filing storage must be a subdirectory of this repo's ignored artifacts/")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_html(data: bytes) -> None:
    prefix = data[:4096].lower()
    if b"<html" not in prefix and b"<!doctype html" not in prefix:
        raise ValueError("Response is not an HTML filing")
    if b"</html>" not in data[-4096:].lower():
        raise ValueError("HTML response has no closing document tag")
    if b"undeclared automated tool" in prefix or b"request rate threshold exceeded" in prefix:
        raise ValueError("SEC access denial page is not a filing")


def store_filing(filing: Filing, data: bytes, directory: Path) -> None:
    filing.validate()
    validate_html(data)
    if len(data) != filing.bytes or hashlib.sha256(data).hexdigest() != filing.sha256:
        raise ValueError(f"Content changed or corrupted: {filing.ticker}")
    target = ignored_store(directory) / filing.filename
    if target.is_symlink():
        raise ValueError("Refusing symlink filing path")
    temporary = target.with_suffix(".download")
    if temporary.is_symlink():
        raise ValueError("Refusing symlink temporary path")
    temporary.write_bytes(data)
    temporary.replace(target)


def load_manifest(path: Path) -> list[Filing]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not 1 <= len(payload["filings"]) <= 2:
        raise ValueError("Expected schema v1 with one or two filings")
    filings = [Filing(**record) for record in payload["filings"]]
    for filing in filings:
        filing.validate()
    if len({filing.url for filing in filings}) != len(filings):
        raise ValueError("Duplicate corpus URLs")
    return filings


def create_manifest(client: SECClient, directory: Path, manifest: Path) -> list[Filing]:
    if manifest.exists():
        raise ValueError("Manifest exists; use reproduce instead of overwriting frozen metadata")
    ignored_store(directory)
    filings = []
    for ticker, cik in COMPANIES.items():
        metadata_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        recent = json.loads(client.fetch(metadata_url, metadata=True))["filings"]["recent"]
        index = recent["form"].index("10-K")
        accession = recent["accessionNumber"][index]
        document = recent["primaryDocument"][index]
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{document}"
        )
        data = client.fetch(url)
        filing = Filing(
            ticker,
            cik,
            "10-K",
            recent["reportDate"][index],
            recent["filingDate"][index],
            accession,
            url,
            hashlib.sha256(data).hexdigest(),
            len(data),
        )
        store_filing(filing, data, directory)
        filings.append(filing)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"schema_version": 1, "filings": [asdict(f) for f in filings]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return filings


def reproduce(filings: list[Filing], directory: Path, fetch: Callable[[str], bytes]) -> None:
    ignored_store(directory)
    for filing in filings:
        store_filing(filing, fetch(filing.url), directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["create", "reproduce", "verify"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "corpus/edgar-manifest.json")
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/edgar-html")
    parser.add_argument("--user-agent")
    args = parser.parse_args()
    if args.mode == "create":
        filings = create_manifest(SECClient(args.user_agent or ""), args.directory, args.manifest)
    else:
        filings = load_manifest(args.manifest)
        if args.mode == "reproduce":
            reproduce(filings, args.directory, SECClient(args.user_agent or "").fetch)
        else:
            directory = ignored_store(args.directory)
            for filing in filings:
                path = directory / filing.filename
                if path.is_symlink():
                    raise ValueError("Refusing symlink filing path")
                data = path.read_bytes()
                validate_html(data)
                if len(data) != filing.bytes or hashlib.sha256(data).hexdigest() != filing.sha256:
                    raise ValueError(f"Local verification failed: {filing.ticker}")
    print(json.dumps({"mode": args.mode, "filings": len(filings), "verified": True}))


if __name__ == "__main__":
    main()
