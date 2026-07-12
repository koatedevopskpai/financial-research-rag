# Financial Research RAG

A production-grade Retrieval-Augmented Generation system purpose-built for financial investment research. Ingests financial documents (annual reports, earnings transcripts, analyst notes), indexes them with **hybrid search** (vector + BM25), reranks with **Azure Semantic Ranker**, and answers complex financial queries via **Azure OpenAI GPT-4o** — with **graceful degradation** at every pipeline step.

> Built to demonstrate the exact skills required for a Senior AI Engineer role: Azure AI Services, RAG pipeline engineering, production-grade Python, resilience patterns, and infrastructure-as-code deployment.

---

## The Problem

Investment research teams spend hours manually searching through thousands of pages of financial documents (annual reports, earnings call transcripts, analyst notes, regulatory filings) to answer specific questions:

- *"What was Apple's EBITDA margin trend over the last 3 fiscal years?"*
- *"Which companies mentioned AI more than 10 times in their Q3 earnings calls?"*
- *"Show me all references to 'share buyback' in Microsoft's 2025 annual report."*

Existing document search tools are either:
- **Keyword-only** (misses semantic queries like "growing free cash flow")
- **AI chatbots** (hallucinate metrics and fabricate citations — unacceptable for finance)
- **Expensive** (requiring dedicated GPU infrastructure for custom models)

## The Solution

**Financial Research RAG** bridges this gap — combining hybrid search precision with LLM-powered reasoning, grounded in strict citation requirements.

| Feature | Capability |
|---------|-----------|
| **Hybrid Search** | Vector (semantic) + BM25 (keyword) — weighted, configurable |
| **Hierarchical Chunking** | Parent chunks for LLM context, child chunks for precision retrieval |
| **Semantic Reranking** | Azure Semantic Ranker re-scores results for 30-40% relevance improvement |
| **Strict Grounding** | LLM must cite every metric by filename — no fabricated numbers |
| **SSE Streaming** | Real-time token streaming via Server-Sent Events |
| **Graceful Degradation** | Every step has a fallback — system never silently fails |
| **Infrastructure as Code** | Terraform provisions all Azure resources — reproducible, auditable |

---

## Architecture

```
                         ┌──────────────────┐
                         │    Client App    │
                         │  (SSE / HTTP)    │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │   FastAPI App    │
                         │  POST /ingest   │
                         │  POST /query    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
            ┌───────▼────────┐        ┌─────────▼────────┐
            │ Ingestion      │        │ Query            │
            │ Pipeline       │        │ Pipeline         │
            │                │        │                  │
            │ Parse → Chunk  │        │ Embed → Search   │
            │ → Embed → Index│        │ → Rerank → Gen   │
            └───────┬────────┘        └─────────┬────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      Azure Services       │
                    │                           │
                    │  ┌─────────────────────┐  │
                    │  │ Azure AI Search     │  │
                    │  │ (Hybrid Index +     │  │
                    │  │  Semantic Ranker)   │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │ Azure OpenAI        │  │
                    │  │ (GPT-4o + Ada-002)  │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │ AKS + ACR           │  │
                    │  └─────────────────────┘  │
                    └───────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Framework** | FastAPI (Python 3.11+, async, typed) |
| **Data Models** | Pydantic v2 — strict mode, validation at every boundary |
| **Vector Store** | Azure AI Search — HNSW indexing, hybrid search, semantic ranker |
| **LLM** | Azure OpenAI — GPT-4o (generation) + text-embedding-ada-002 (embeddings) |
| **Containerisation** | Docker (multi-stage) + docker-compose (local dev) |
| **Orchestration** | Kubernetes (AKS) — deployment manifests included |
| **Infrastructure** | Terraform — Azure OpenAI, AI Search, ACR, AKS |
| **Observability** | OpenTelemetry — structured logging (structlog), distributed tracing |
| **CI/CD** | GitHub Actions — lint (ruff), type-check (mypy), test (pytest), build |
| **Auth** | Azure AD Managed Identity — no API keys in config |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hierarchical chunking** | Small chunks (512 tokens) for retrieval precision; large parent chunks (2048 tokens) provide LLM context — avoids "lost in the middle" |
| **Hybrid search** | Financial queries use exact terms ("ticker: AAPL") where keyword beats vector; semantic search handles conceptual queries |
| **Semantic Ranker** | Managed cross-encoder — no GPU hosting, no model maintenance, 30-40% relevance improvement |
| **Circuit breaker on AOAI** | Prevents thundering herd under rate limits — 5 failures, 30s recovery |
| **Graceful degradation** | Every step has a defined fallback — never a 500 response |
| **No fallback embedding** | Swapping models mid-pipeline changes vector space — degrade to keyword instead |
| **Single FastAPI service** | No inter-service latency, simpler deployment, single trace context |
| **Strict citation rules** | Every quantitative claim must cite source filename — no fabrications |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Azure subscription with access to:
  - Azure OpenAI (GPT-4o + text-embedding-ada-002 deployments)
  - Azure AI Search (Standard tier — required for vector search)
- Azure CLI (authenticated with `az login`)

### Local Development

```bash
# Clone
git clone <repo-url>
cd financial-research-rag

# Create virtualenv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file with your Azure credentials
cat > .env << EOF
FRR_AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
FRR_AZURE_SEARCH_ENDPOINT=https://<your-service>.search.windows.net
FRR_AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
FRR_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
FRR_AZURE_SEARCH_INDEX_NAME=financial-documents
FRR_APP_LOG_LEVEL=DEBUG
EOF

# Run
make dev
```

### Docker

```bash
make docker-run
```

### Run Tests

```bash
make test
```

### Lint & Typecheck

```bash
make check
```

---

## API Endpoints

### `POST /ingest`

Upload a financial document for processing.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@annual_report_2025.pdf"
```

**Response:**
```json
{
  "message": "Document indexed successfully",
  "results": [...],
  "total_documents": 1,
  "total_chunks_indexed": 42,
  "processing_time_ms": 3200
}
```

### `POST /query`

Ask a question against the indexed corpus.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple EBITDA margin trend?", "stream": true}'
```

**Streaming response (SSE):**
```
event: token
data: {"token": "Based"}

event: token
data: {"token": " on"}

event: token
data: {"token": " the"}

event: token
data: {"token": " retrieved"}

...

event: done
data: {"answer": "...", "sources": [...], "total_sources_retrieved": 10}
```

---

## Project Structure

```
financial-research-rag/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Pydantic Settings — all Azure config
│   ├── models/
│   │   ├── documents.py         # Document, Chunk, SourceMetadata
│   │   └── requests.py          # QueryRequest, QueryResponse, ErrorResponse
│   ├── ingestion/
│   │   ├── chunker.py           # Hierarchical chunker with semantic boundaries
│   │   ├── embedder.py          # Azure OpenAI embedding with circuit breaker
│   │   └── indexer.py           # Azure AI Search index creation + upload
│   ├── retrieval/
│   │   ├── search.py            # Hybrid search (vector + BM25)
│   │   └── reranker.py          # Semantic Ranker reranking
│   ├── generation/
│   │   ├── rag_chain.py         # Retrieve → augment → generate
│   │   └── prompts.py           # Grounded prompt templates
│   ├── pipelines/
│   │   ├── document_pipeline.py # End-to-end ingestion orchestration
│   │   └── query_pipeline.py    # End-to-end query orchestration
│   └── utils/
│       ├── logging.py           # Structured JSON logging (structlog)
│       ├── monitoring.py        # OpenTelemetry tracing
│       └── error_handling.py    # Typed exceptions + consistent error responses
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile           # Multi-stage Python 3.11 slim image
│   │   └── docker-compose.yml   # Local development environment
│   ├── k8s/
│   │   ├── deployment.yaml       # AKS deployment + service
│   │   └── configmap.yaml       # Non-secret configuration
│   └── terraform/
│       ├── main.tf              # Azure resource provisioning
│       ├── variables.tf         # Tunable parameters
│       └── outputs.tf           # Endpoints as outputs
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_chunker.py          # Chunk boundary correctness
│   ├── test_retrieval.py        # Hybrid search + reranker tests
│   └── test_pipelines.py        # End-to-end pipeline orchestration
├── docs/
│   ├── architecture.md          # Full system architecture
│   └── decisions.md             # Architecture Decision Records (8 ADRs)
├── .github/workflows/
│   └── ci.yml                   # Lint → typecheck → test → build
├── pyproject.toml               # Project metadata + tool configs
├── requirements.txt             # Pinned dependencies
├── Makefile                     # Dev, test, lint, docker, terraform targets
└── README.md
```

---

## Graceful Degradation Matrix

| Component Failure | Degraded Behaviour |
|-------------------|-------------------|
| Azure OpenAI (embedding) | Falls back to keyword-only (BM25) search |
| Azure OpenAI (generation) | Returns extractive summary from top chunks |
| Azure AI Search (down) | Returns `503` with `degraded_mode: true` |
| Circuit breaker (AOAI) | Short-circuits after 5 failures, auto-retries after 30s |
| Document parsing fails | Returns `IndexingResult(success=False)` with error message |
| Semantic Ranker (down) | Falls back to hybrid search scores |

---

## Resilience Patterns

| Pattern | Implementation |
|---------|---------------|
| **Retry with backoff** | `tenacity` — exponential backoff (2s → 4s → 8s → ...), max 3 attempts |
| **Circuit breaker** | Custom implementation — 5 consecutive failures → open 30s → half-open → closed |
| **Graceful degradation** | Every pipeline step has a defined fallback strategy |
| **Structured error handling** | Typed exceptions (RAGError hierarchy) with correlation IDs |
| **Distributed tracing** | OpenTelemetry — correlation IDs across ingestion → index → query |
| **Health probes** | K8s liveness + readiness — pod restarts if degraded too long |

---

## Infrastructure (Deploy to Azure)

```bash
# Provision Azure resources
cd infrastructure/terraform
terraform init
terraform plan
terraform apply

# Build and push Docker image
az acr login --name <acr-name>
docker build -t <acr-name>.azurecr.io/financial-research-rag:latest -f infrastructure/docker/Dockerfile .
docker push <acr-name>.azurecr.io/financial-research-rag:latest

# Deploy to AKS
az aks get-credentials --resource-group <rg-name> --name <aks-name>
kubectl apply -f infrastructure/k8s/
```

---

## Testing Strategy

| Test Layer | What It Covers |
|------------|----------------|
| **Unit tests** (test_chunker.py) | Chunk boundary correctness, token count accuracy, metadata preservation |
| **Unit tests** (test_retrieval.py) | Hybrid search construction, filter building, reranker scoring |
| **Integration tests** (test_pipelines.py) | Pipeline orchestration, error handling, degraded mode paths |

Run: `make test` or `pytest tests/ -v --cov=app --cov-report=term-missing`

---

## Why This Project Matters for an AI Engineer Role

This project demonstrates exactly what the role demands:

1. **Azure AI Services** — Azure OpenAI (GPT-4o, embeddings), Azure AI Search (hybrid index, semantic ranker), Azure AD auth
2. **RAG Pipeline Engineering** — Chunking strategies, hybrid search, reranking, context window management, prompt grounding
3. **Resilience Engineering** — Circuit breaker, retry with backoff, graceful degradation at every step
4. **Production Python** — FastAPI, Pydantic v2 strict, async, type hints, structured logging, OpenTelemetry tracing
5. **Infrastructure as Code** — Terraform, Docker multi-stage, Kubernetes manifests, CI/CD
6. **Domain-Driven Design** — ADR documentation, opinionated architecture, measurable trade-offs

---

## License

MIT
