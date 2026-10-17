# Financial Research RAG — experimental reference implementation

This repository contains Azure AI Search / Azure OpenAI integration code, a
FastAPI API, hierarchical chunking, citation-oriented prompt assembly and tests.
It does not currently establish a deployed production financial research system,
measured retrieval/answer quality. Do not present
it as commercial banking experience or a validated financial research product.

## Supported preparation and tests (S1)

Python 3.11 is the CI target. From the repository root:

```sh
pip install -r requirements-ci.lock
pip install --no-deps -e .
python -m app.tokenizer_assets --prepare
ruff check app/ tests/
mypy app/ tests/
python -m pytest tests/ -v --cov=app --cov-report=xml --cov-report=term-missing
```

The lock pins the CI environment across supported platform markers. Requirements
and pyproject.toml retain declared dependency ranges; regenerate/revalidate the
lock deliberately. Tokenizer preparation is a separate network step: bounded
download, expected SHA-256, ignored local cache. Runtime requires that asset and
fails clearly if missing/corrupt; it does not silently download or substitute a
fake tokenizer. Tests block external Python socket/DNS calls and use real cached
tokenization with explicit SDK/client mocks. This is not a security sandbox.

The Docker build prepares the same cache and uses the lock. No Azure credentials
are required to import the app or run these tests. Missing Azure endpoints fail
clearly at service construction; health reports initialization flags and degraded
status when pipelines are unavailable. A liveness response is not proof of live
Azure connectivity. Actual Azure integration/retrieval quality remains unverified.

## Ingestion boundary

Only UTF-8 TXT and basic HTML extraction are implemented. PDF/DOCX/XLSX fail
explicitly; the old placeholder text path is removed. HTML is still a basic tag
stripper, not a layout/table-aware parser. S2 adds a [two-filing SEC EDGAR manifest](docs/S2.md), with actual downloads and
fresh hash reproduction. Filing text remains ignored/local only. Retrieval/answer
evaluation is S5; no benchmark or financial answer quality is established. B01 remains deferred.

## Repairs and limits

S1 incorporates the handed-over import/schema fixes, repairs async test calls and
embedding/circuit-breaker awaits, uses refreshing token callbacks rather than an
AccessToken object, aligns Azure Search SDK schema/upload types, preserves missing
vectors rather than inventing zero vectors, and makes degraded generation/time
metadata explicit. Raw SDK/JSON dictionaries remain dynamically typed at their
external boundaries. Strict mypy/ruff are not disabled or narrowed to hide errors.

Tests and a local container smoke establish only their recorded scope. The live
Azure generation/search paths, concurrency limits of the simple circuit breaker,
semantic ranking behaviour, retries under real provider faults, corpus quality,
financial correctness and production readiness still require evidence. See
[S1](docs/S1.md) for exact revisions, commands, results and residual limits.
