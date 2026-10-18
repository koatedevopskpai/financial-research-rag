# Evidence index — Financial Research RAG

A reviewer's route through this repository. Every claim points at code, tests, or a
dated receipt in [`evidence/`](evidence/). No claim is asserted without a location.

**Five-minute path:** this file → [what it is](#what-it-is) →
[the strongest measured result](#the-strongest-measured-result) →
[how to reproduce](#how-to-reproduce) →
[what this does not establish](#what-this-does-not-establish).

---

## What it is

An **experimental** Retrieval-Augmented Generation reference implementation for
financial documents: Azure AI Search / Azure OpenAI integration, a FastAPI service,
hierarchical chunking, citation-oriented prompt assembly, and a separate
retrieval/citation/abstention evaluation harness.

The README states the boundary in its first paragraph, and this file repeats it because
it is the most likely thing to be misread: this repository **does not** establish a
deployed production financial research system, and it **does not** establish measured
retrieval or answer quality against real financial questions.

| | |
|---|---|
| Scope | **Public reference build.** Self-funded; no customer, no deployment currently live |
| Author | Koate Kpai — https://github.com/koatedevopskpai |
| Entry point | `app/`, `tests/`, `docs/` |
| Revision described | `75a98ab` |
| CI | **green** on `main`, 18 October 2026 |

## My contribution

| Area | Source and decision |
|---|---|
| Hardening pass — import/schema fixes, async/await repair, refreshing token callbacks, Azure Search SDK schema/upload alignment, missing vectors preserved rather than zero-filled, explicit degraded generation metadata | [`docs/S1.md`](docs/S1.md), [`evidence/S1.md`](evidence/S1.md) |
| Ingestion boundary made explicit — only UTF-8 TXT and basic HTML; PDF/DOCX/XLSX **fail explicitly** instead of returning a text stub | [`README.md`](README.md), [`docs/S1.md`](docs/S1.md) |
| Reproducible environment — a CI lock, a separate, SHA-256-checked tokenizer preparation step that fails closed | [`requirements-ci.lock`](requirements-ci.lock), `app/tokenizer_assets` |
| SEC EDGAR corpus with hash-verified downloads | [`docs/S2.md`](docs/S2.md), [`evidence/S2.md`](evidence/S2.md) |
| Separate retrieval, quote-support and abstention checks with three rejected regressions | [`docs/S5.md`](docs/S5.md), [`evidence/S5.md`](evidence/S5.md) |

---

## The strongest measured result

**The S5 fault-injection harness.** It runs a tiny local BM25/extractive adapter over
**eight authored retrieval cases** and checks that the gate *rejects* each injected
fault rather than passing it:

| Run | Recall@5 | MRR@5 | Supported quotes | Answer contract | Abstention acc. | Gate |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1.000 | 1.000 | 8/8 | 1.000 | 1.000 | **True** |
| retrieval fault | 0.000 | 0.000 | 0/0 | 0.200 | 0.200 | **False** |
| citation fault | 1.000 | 1.000 | 0/8 | 0.200 | 1.000 | **False** |
| abstention fault | 1.000 | 1.000 | 8/10 | 0.800 | 0.800 | **False** |

The value here is the rejection behaviour, not the baseline row: a gate that cannot
fail is not evidence. Eight cases is a regression check, not a benchmark.

**Recorded engineering receipts** (Linux, CPython 3.11 — the CI target):

| Receipt | Scope recorded |
|---|---|
| [`evidence/S1.md`](evidence/S1.md) | ruff, strict mypy (30 files), **23 tests**, Docker image build and an unconfigured-container smoke test. 67% coverage |
| [`evidence/S2.md`](evidence/S2.md) | Two actual SEC filings downloaded twice; exact bytes and SHA-256 verified; **41 tests**, mypy (32 files), lint |
| [`evidence/S5.md`](evidence/S5.md) | **49 tests**, strict mypy (35 files), lint, four real CLI exit/report checks |

**Provenance note.** These are the committed receipts, **not re-run for this index** —
they are Linux/CPython 3.11 and this index was written on Windows/CPython 3.14. The
current `main` CI is green (18 October 2026), which is a separate, weaker statement.

### A claim that was corrected

This repository previously described itself as **production-grade**, including in
`docs/architecture.md` and the package docstring. It establishes no such thing. The
wording was corrected to "experimental reference implementation" rather than softened
(commit `75a98ab`). Correcting a claim downward is part of the evidence trail.

---

## How to reproduce

Python **3.11** is the CI target. From the repository root:

```sh
pip install -r requirements-ci.lock
pip install --no-deps -e .
python -m app.tokenizer_assets --prepare     # bounded download, SHA-256 checked, fails closed
ruff check app/ tests/
mypy app/ tests/
python -m pytest tests/ -v --cov=app --cov-report=term-missing
```

Tokenizer preparation is a **separate network step**. The runtime requires that asset
and fails clearly if it is missing or corrupt; it does not silently download or
substitute a fake tokenizer. Tests block external socket/DNS calls — a process-local
guard, **not** a security sandbox.

**No Azure credentials are required** to import the app or run these tests. Missing
Azure endpoints fail clearly at service construction; health reports initialization
flags and a degraded status when pipelines are unavailable. A liveness response is
**not** proof of live Azure connectivity.

**Live, billable, and not part of this evidence.** The Azure generation and search
paths were not exercised for any receipt above.

---

## One success trace, one failure

**Success.** The S1 receipt records a real container build and an unconfigured-container
smoke test passing — so the image runs and degrades explicitly rather than crashing,
without any Azure credentials present.

**Failure, blocked by design.** A missing or hash-mismatched tokenizer asset stops the
run instead of substituting a fake encoder. Unsupported document types (PDF/DOCX/XLSX)
raise rather than returning a placeholder string. Both are the correct outcome; a
silently degraded ingestion path would be worse than a visible failure.

**Failure the harness catches.** Each of the three injected faults above is *rejected*
by the gate. The harness exists specifically so that a regression cannot pass quietly.

---

## Decisions and trade-offs

| Decision | Where |
|---|---|
| Fail closed on a missing/corrupt tokenizer asset; never substitute | [`README.md`](README.md), `app/tokenizer_assets` |
| Remove the placeholder PDF/DOCX/XLSX text path rather than pretend to parse | [`README.md`](README.md), [`docs/S1.md`](docs/S1.md) |
| Preserve missing vectors instead of inventing zero vectors | [`docs/S1.md`](docs/S1.md) |
| Separate retrieval, citation and abstention checks on a tiny local adapter | [`docs/S5.md`](docs/S5.md) |
| SEC EDGAR HTML as the first corpus; hash-verified, text not distributed | [`docs/S2.md`](docs/S2.md) |

---

## What this does not establish

- **Deployed financial research.** No live system, no customer, no uptime claim.
- **Financial answer quality.** S5 uses eight authored cases and a local
  BM25/extractive adapter. It is exact-quote support, **not** semantic entailment or
  answer relevance, and it says nothing about the live pipeline.
- **Live Azure behaviour.** No Azure call is recorded in any receipt: not retrieval,
  not generation, not semantic ranking, not retries under real provider faults.
- **Concurrency or resilience at scale.** The circuit breaker is simple and its
  concurrency limits are unmeasured.
- **Parse fidelity.** HTML extraction is a basic tag stripper, not a layout- or
  table-aware parser.
- **A benchmark.** B01 remains deferred; eight cases and 49 tests are not one.

## Known weaknesses

- Coverage on the S1 receipt is **67%** and is not gated.
- Azure SDK/JSON dictionaries remain dynamically typed at their external boundaries;
  strict typing stops at that edge by design, not by omission.
- The corpus is two filings and is deliberately **not distributed** in the repository.
- Upstream EDGAR access policies could block future fetches; the SHA-256 check fails
  closed if a document changes.
