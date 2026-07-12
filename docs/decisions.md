# Architecture Decision Records (ADR)

This document records the key architectural decisions made during the design of the Financial Research RAG system. Each entry follows the ADR format: context, decision, consequences.

---

## ADR-001: Hierarchical Chunking (Parent/Child)

**Status:** Accepted

**Context:** Financial documents contain dense information at varying granularities. A chunk that is optimal for embedding (small, focused) may lack the surrounding context needed by the LLM to produce an accurate answer. A chunk that is optimal for LLM context (large, comprehensive) loses retrieval precision.

**Decision:** Implement hierarchical chunking with two levels:
- **Parent chunks** (~2048 tokens) — stored in the index but NOT used for vector search. Retrieved at query time by joining on `parent_chunk_id` to provide the LLM with surrounding context.
- **Child chunks** (~512 tokens) — embedded and used for vector search. These are the retrieval granularity.

**Consequences:**
- + Precise retrieval (small chunks pin-point specific passages)
- + LLM gets full context (parent chunks prevent lost-in-the-middle)
- + No additional latency (parent lookup is a constant-time index join)
- - More storage (2x chunks for the same document)
- - More embedding cost (child chunks only, so marginal increase)

---

## ADR-002: Hybrid Search (Vector + BM25)

**Status:** Accepted

**Context:** Financial queries are often term-heavy ("ticker: AAPL", "EBITDA margin 2025", "Fiscal Year 2025 Q3"). Pure vector search can miss these exact term matches. Pure BM25 cannot capture semantic similarity ("companies with growing free cash flow").

**Decision:** Use Azure AI Search's built-in hybrid search, combining:
- Vector search on chunk embeddings (cosine similarity)
- Full-text BM25 search on chunk content
- Weighted scoring (configurable via `retrieval_vector_weight`, default 0.5)

**Consequences:**
- + Catches both semantic and exact-term queries
- + No additional infrastructure (built into Azure AI Search)
- + Configurable weighting per use case
- - Slightly higher query latency (both search paths execute)
- - Scoring calibration needed between vector and BM25

---

## ADR-003: Azure Semantic Ranker Instead of Custom Cross-Encoder

**Status:** Accepted

**Context:** After hybrid search returns top-K candidates, we need to rerank them to ensure the most relevant results are surfaced. Options: host a custom cross-encoder (e.g., BERT) or use Azure's managed Semantic Ranker.

**Decision:** Use Azure AI Search Semantic Ranker — a managed cross-encoder fine-tuned for search.

**Consequences:**
- + No GPU hosting cost for custom cross-encoder
- + No model maintenance, versioning, or fine-tuning
- + Improves top-K relevance by 30-40% over vector alone
- - Pay-per-query cost (negligible for research team volume)
- - Locked to Azure's model (no custom domain fine-tuning)

---

## ADR-004: Single FastAPI Service (Not Microservices)

**Status:** Accepted

**Context:** The system handles ingestion and query workloads. For investment research teams (not internet-scale), these can be separate services or a single deployable unit.

**Decision:** Single FastAPI service exposing both `/ingest` and `/query` endpoints.

**Consequences:**
- + No inter-service latency (no HTTP hops between ingestion → query)
- + Simpler deployment (one Docker image, one K8s service)
- + Single tracing context (easier debugging)
- - Ingestion and query share resources (could be separated later if needed)

---

## ADR-005: Circuit Breaker for Azure OpenAI Calls

**Status:** Accepted

**Context:** Azure OpenAI has rate limits and can return HTTP 429 (Too Many Requests) under load. Without protection, repeated retries would compound the problem (thundering herd).

**Decision:** Implement a circuit breaker pattern:
- CLOSED: normal operation
- OPEN: after 5 consecutive failures, reject all calls for 30s
- HALF_OPEN: after timeout, allow one test call

**Consequences:**
- + Prevents cascading failures
- + Gives Azure time to recover under rate limits
- + Degraded mode kicks in gracefully (keyword search)
- - Added complexity (state management)
- - 30s recovery may be too aggressive for transient blips

---

## ADR-006: No Fallback Embedding Model

**Status:** Accepted

**Context:** If Azure OpenAI's embedding API is unavailable, should we fall back to a local embedding model (e.g., sentence-transformers)?

**Decision:** Do not fall back to a different embedding model. Instead, degrade to keyword-only (BM25) search.

**Consequences:**
- + Consistent vector space (swapping models mid-pipeline changes the vector space and produces unreliable results)
- + Simpler code (no local model hosting)
- - Lower retrieval quality during degraded mode
- - Financial research values consistency over fallback quality

---

## ADR-007: Strict Citation Requirements for LLM Output

**Status:** Accepted

**Context:** Financial hallucinations are costly. A fabricated metric could lead to a wrong investment decision. The LLM must cite every quantitative claim.

**Decision:** Enforce strict citation rules in the system prompt:
- Every quantitative claim must cite the source filename
- If context doesn't contain the answer, the model must say so
- Contradictory information must present both sides

**Consequences:**
- + Verifiable answers (analysts can check sources)
- + Reduced hallucination risk
- - Longer responses (citation overhead)
- - Requires prompt engineering to enforce consistently

---

## ADR-008: Graceful Degradation at Every Pipeline Step

**Status:** Accepted

**Context:** External dependencies (Azure OpenAI, Azure AI Search) can fail independently. A single failure should not crash the entire pipeline.

**Decision:** Every pipeline step has a defined degraded mode:
| Step | Failure | Degraded Behaviour |
|------|---------|-------------------|
| Embedding | AOAI unavailable | Keyword-only search |
| Retrieval | AI Search unavailable | Return 503 with degraded_mode |
| Reranking | Semantic ranker unavailable | Use hybrid search scores |
| Generation | AOAI unavailable | Extractive summary from chunks |

**Consequences:**
- + System never returns 500 for downstream failures
- + Users always get a response (even if degraded)
- + Clear degraded_mode flag for client to interpret
- - More code paths to test
