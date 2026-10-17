"""Explicit checksummed preparation; runtime never downloads tokenizer assets silently."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

import tiktoken

URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"


def cache_path() -> Path:
    directory = Path(
        os.environ.get(
            "TIKTOKEN_CACHE_DIR", str(Path(__file__).parents[1] / "artifacts/tokenizer-cache")
        )
    )
    os.environ["TIKTOKEN_CACHE_DIR"] = str(directory.resolve())
    return directory / hashlib.sha1(URL.encode()).hexdigest()


def prepare() -> Path:
    path = cache_path()
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
        return path
    with urlopen(URL, timeout=30) as response:
        data = response.read(4_000_001)
    if len(data) > 4_000_000 or hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("tokenizer asset checksum/size mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


def get_tokenizer() -> tiktoken.Encoding:
    path = cache_path()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError(
            "Tokenizer asset missing/corrupt: run python -m app.tokenizer_assets --prepare"
        )
    return tiktoken.get_encoding("cl100k_base")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()
    print("Tokenizer asset prepared and SHA-256 verified:", prepare())
