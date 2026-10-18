# Financial Research RAG — Architecture

## System Overview

An experimental Retrieval-Augmented Generation reference implementation for financial documents. Ingests financial documents (annual reports, earnings transcripts, analyst notes), indexes them with hybrid search, and answers complex financial queries via natural language.

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│   Client     │────▶│   FastAPI App    │────▶│  Azure AI     │
│  (SSE/HTTP)  │     │  (main.py)       │     │  Search       │
└──────────────┘     └──────┬───────────┘     └───────────────┘
                            │
                    ┌───────┴───────────┐
                    │   Query Pipeline  │
                    │  ┌─────────────┐  │
                    │  │ Embed Query  │  │
                    │  └──────┬──────┘  │
                    │  ┌──────┴──────┐  │
                    │  │Hybrid Search│  │
                    │  └──────┬──────┘  │
                    │  ┌──────┴──────┐  │
                    │  │  Reranker    │  │
                    │  └──────┬──────┘  │
                    │  ┌──────┴──────┐  │
                    │  │ RAG Chain   │──┼──▶ Azure OpenAI (GPT-4o)
                    │  └─────────────┘  │
                    └──────────────────┘
```

## Core Components

### 1. FastAPI Application (`app/main.py`)
Single FastAPI service exposing two endpoints:
- `POST /ingest` — Upload financial documents
- `POST /query` — Ask questions (supports SSE streaming)

### 2. Ingestion Pipeline (`app/pipelines/document_pipeline.py`)
**Parse → Chunk → Embed → Index**

**Parser** (`app/ingestion/`): Extracts text from PDF, DOCX, XLSX, TXT, HTML. Production paths use Azure Document Intelligence.

**Chunker** (`app/ingestion/chunker.py`): Hierarchical chunking with semantic boundary detection. Produces parent chunks (~2048 tokens) for LLM context and child chunks (~512 tokens) for precise retrieval. Respects financial document section headers as hard boundaries.

**Embedder** (`app/ingestion/embedder.py`): Batches child chunks (up to 16 at a time) and generates embeddings via Azure OpenAI `text-embedding-ada-002`. Features circuit breaker protection against sustained API failures.

**Indexer** (`app/ingestion/indexer.py`): Creates Azure AI Search index with an explicit schema (vector + non-vector fields, HNSW algorithm, semantic ranker config). Uploads documents in batches of 1000.

### 2. Query Pipeline (`app/pipelines/query_pipeline.py`)
**Embed → Search → Rerank → Generate**

### 3. Generation (`app/generation/rag_chain.py`)
Retrieves parent + child chunks, builds a grounded prompt with strict citation rules, generates an answer via Azure OpenAI GPT-4o. Supports SSE streaming. Falls back to extractive summarization if the LLM is unavailable.

## Key Architectural Decisions

See `docs/decisions.md` for the full ADR set.

## Infrastructure

| Component | Azure Service | Purpose |
|-----------|---------------|---------|
| Compute | AKS (Azure Kubernetes Service) | Container orchestration, autoscaling |
| Image registry | ACR (Azure Container Registry) | Docker image storage |
| Embedding + generation | Azure OpenAI (GPT-4o, Ada-002) | LLM and vector embeddings |
| Vector store + search | Azure AI Search (Standard) | Hybrid search with semantic ranker |
| Authentication | Azure AD Managed Identity | No API keys in config |
| Observability | Azure Monitor + OpenTelemetry | Distributed tracing, metrics |

## Data Flow: Ingestion

```
Client uploads PDF
       │
       ▼
FastAPI validates file type
       │
       ▼
Document pipeline parses text
       │
       ▼
Chunker creates hierarchical chunks (parent + child)
       │
       ▼
Embedder generates vector embeddings for child chunks
       │
       ▼
Indexer pushes chunks to Azure AI Search
       │
       ▼
Response returned with indexing summary
```

## Data Flow: Query

```
Client sends: "What was AAPL's EBITDA margin in 2025?"
       │
       ▼
Query pipeline embeds the question
       │
       ▼
Hybrid search (vector + BM25) retrieves top-K candidates
       │
       ▼
Semantic Ranker reranks candidates by relevance
       │
       ▼
RAG chain builds prompt with retrieved context
       │
       ▼
LLM generates grounded answer (streaming or full)
       │
       ▼
Response returned with citations
```

## Degraded Mode Matrix

| Component Failure | Behaviour |
|-------------------|-----------|
| Azure OpenAI (embedding) | Falls back to keyword-only (BM25) search |
| Azure OpenAI (generation) | Returns extractive summary from top chunks |
| Azure AI Search | Returns 503 with degraded_mode flag |
| Circuit breaker open | Short-circuits after N failures, retries after timeout |
