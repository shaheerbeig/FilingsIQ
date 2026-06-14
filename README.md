<p align="center">
  <img src="docs/banner.svg" alt="FilingsIQ — Retrieval-Augmented Intelligence for SEC Filings" width="100%">
</p>

<h1 align="center">FilingsIQ</h1>
<p align="center"><strong>Retrieval-Augmented Intelligence for SEC Filings</strong></p>
<p align="center">Ask a company's annual report anything — and get fast, grounded, citation-backed answers, with zero hallucination.</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-0c8a5b">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-0c8a5b">
  <img alt="Vector DB" src="https://img.shields.io/badge/Vector%20DB-Chroma-0c8a5b">
  <img alt="LLM" src="https://img.shields.io/badge/LLM-OpenAI%20GPT--4o-0c8a5b">
  <img alt="Status" src="https://img.shields.io/badge/status-working-0c8a5b">
</p>

---

## 📌 What is this?

A company's **10-K** (annual report) is ~120 pages of dense financial and legal text. Finding one fact means either reading it all, or asking a general chatbot that **makes numbers up**.

**FilingsIQ** is a full **Retrieval-Augmented Generation (RAG)** system that fixes this. You ask a plain-English question; it finds the exact passages in the filing, reads them, and writes a **grounded answer with citations** — and honestly says *"I can't find that"* when the document doesn't contain the answer.

It currently ships with **Apple Inc.'s FY2024 Form 10-K** indexed and ready to query.

| | |
|---|---|
| ![Home](docs/ui-home.png) | ![Answer](docs/ui-answer.png) |
| *Ask box with example questions* | *Grounded answer with inline citations and sources* |

---

## ✨ Features

- **Grounded answers, not guesses** — every answer is built only from passages retrieved from the filing.
- **Inline citations** — each answer cites the exact chunks `[1] [2]` and lists their section + page.
- **Refuses to hallucinate** — when the filing doesn't cover something, it says so instead of inventing facts.
- **Reasons over numbers** — computes percentages, sums and differences from figures in the document.
- **Two-stage retrieval** — fast vector search casts a wide net; an LLM reranker re-orders for true relevance.
- **Self-evaluating** — a graded test set scores the pipeline objectively (no eyeballing).
- **Clean web UI + JSON API** — one FastAPI app serves both, on a single origin (no CORS setup).

---

## 🏗️ Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Pipeline architecture" width="100%">
</p>

The system has two halves:

- **Ingest (offline, once per document):** turn a PDF into a searchable vector index.
- **Answer (live, per question):** turn a question into a grounded, cited answer.

### The 8 stages

| # | Stage | What it does | Key tech |
|---|-------|--------------|----------|
| 1 | **Parse** | Break the PDF into clean, typed elements (titles, paragraphs, tables) | `unstructured`, `PyMuPDF`, YOLOX layout model |
| 2 | **Chunk** | Group elements into ~400-token passages, preserving section breadcrumbs | `tiktoken` (`cl100k_base`) |
| 3 | **Embed** | Turn each passage into a 1536-dimension meaning vector | OpenAI `text-embedding-3-small` |
| 4 | **Store** | Save vectors + text + metadata in a searchable index | Chroma (HNSW, cosine) |
| 5 | **Retrieve** | Embed the question, fetch the nearest 20 passages | Chroma vector search |
| 6 | **Rerank** | An LLM re-scores each passage against the question, keeps the best 5 | OpenAI `gpt-4o-mini` (parallel) |
| 7 | **Generate** | Write a grounded, cited answer from the top passages | OpenAI `gpt-4o` |
| 8 | **Evaluate** | Score the whole pipeline against a ground-truth test set | LLM-as-judge + retrieval metrics |

### Request flow

```mermaid
sequenceDiagram
    participant U as Investor (browser)
    participant API as FastAPI · /api/ask
    participant R as Retriever
    participant K as Reranker
    participant G as Generator
    participant DB as Chroma

    U->>API: POST { question }
    API->>R: retrieve(question)
    R->>DB: nearest 20 by cosine similarity
    DB-->>R: 20 candidate chunks
    R-->>API: candidates
    API->>K: rerank(question, candidates)
    Note over K: scores all 20 in parallel,<br/>keeps top 5
    K-->>API: top 5 chunks
    API->>G: generate(question, top 5)
    Note over G: "answer ONLY from context,<br/>cite sources, else refuse"
    G-->>API: grounded answer + citations
    API-->>U: { answer, sources }
```

---

## 🧰 Tech stack

| Layer | Technology |
|-------|------------|
| **Parsing** | `unstructured` (+ YOLOX layout detection), `PyMuPDF` |
| **Tokenization** | `tiktoken` |
| **Embeddings** | OpenAI `text-embedding-3-small` (1536-d) |
| **Vector store** | Chroma (local, persistent, HNSW index, cosine distance) |
| **Reranker / Generator / Judge** | OpenAI `gpt-4o-mini` / `gpt-4o` / `gpt-4o` |
| **Config** | Pydantic v2 + `pydantic-settings` (YAML + env + `.env`) |
| **Logging** | `loguru` |
| **API** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML / CSS / JS (no build step) |

---

## 📂 Project structure

```
rag-system/
├── api/
│   ├── main.py              # FastAPI app: /api/ask, /api/health, serves the UI
│   └── static/index.html    # Investor frontend (single page)
├── src/
│   ├── pipeline.py          # RAGPipeline: retrieve → rerank → generate
│   ├── parsing/             # Stage 1 — PDF → elements
│   ├── chunking/            # Stage 2 — elements → chunks
│   ├── embeddings/          # Stage 3 — chunks → vectors
│   ├── storage/             # Stage 4 — Chroma vector store
│   ├── retrieval/           # Stage 5 — question → nearest chunks
│   ├── reranking/           # Stage 6 — LLM re-scoring
│   ├── generation/          # Stage 7 — grounded answer
│   ├── evaluation/          # Stage 8 — metrics + LLM judge
│   └── common/              # config, paths, logging
├── config/default.yaml      # All tunable settings
├── eval/testset.json        # Ground-truth Q&A for evaluation
├── data/                    # raw PDFs, parsed, chunks, embeddings, vector_store
└── requirements.txt
```

Each stage follows the same house style: a **config block** → a **Pydantic data model** → a **logic module**.

---

## 🚀 Getting started

### 1. Prerequisites

- Python 3.11
- An OpenAI API key

### 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...your-key...
```

All other settings live in `config/default.yaml` and can be overridden via environment
variables (e.g. `GENERATION__MODEL=gpt-4o-mini`).

### 4. Run the app

```bash
uvicorn api.main:app --port 8500
```

Open **http://127.0.0.1:8500** and start asking questions.

> The Apple 10-K index ships pre-built in `data/vector_store/`, so the app works out of the box —
> no ingest needed.

---

## 🔌 API reference

### `POST /api/ask`

```bash
curl -X POST http://127.0.0.1:8500/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Apple return cash to shareholders?"}'
```

```json
{
  "question": "How does Apple return cash to shareholders?",
  "answer": "Apple returns cash to shareholders primarily through share repurchase programs and the payment of dividends ... [1]",
  "sources": [
    {
      "number": 1,
      "chunk_id": "chunk_00064",
      "section": "... > Capital Return Program",
      "pages": "p29-29"
    }
  ],
  "model": "gpt-4o"
}
```

### `GET /api/health`

```json
{ "status": "ok", "vectors": 263 }
```

### Deep links

`GET /?q=<url-encoded question>` opens the UI and runs the question automatically — handy for sharing.

---

## 📊 Evaluation

Quality is measured, not eyeballed. `eval/testset.json` holds ground-truth questions paired with
their known answers and the chunk(s) that prove them. The evaluator runs the full pipeline on each
and scores it two ways:

- **Retrieval (objective):** did the correct chunk land in the top-k, and how high? → `hit-rate`, `MRR`
- **Answer (LLM-judged):** does it match the reference, and is every claim grounded? → `correctness`, `faithfulness`

**Latest run:**

| Metric | Score |
|--------|-------|
| Retrieval hit-rate @5 | **100%** |
| Retrieval MRR | **1.00** |
| Answer correctness | **98%** |
| Answer faithfulness | **100%** |

---

## ⚙️ Configuration highlights

All in `config/default.yaml`:

| Key | Meaning | Default |
|-----|---------|---------|
| `embedding.model` | Embedding model | `text-embedding-3-small` |
| `storage.distance` | Vector distance metric | `cosine` |
| `rerank.candidate_k` | How many chunks retrieval fetches before reranking | `20` |
| `rerank.top_n` | How many chunks survive reranking | `5` |
| `rerank.max_workers` | Concurrent rerank calls (latency) | `10` |
| `generation.model` | Answer-writing model | `gpt-4o` |
| `generation.temperature` | `0.0` = deterministic, grounded | `0.0` |

---

## 🧭 Notes & limitations

- Indexed for **one document** (Apple FY2024 10-K). The pipeline is document-agnostic; ingesting a new
  filing is a matter of running stages 1–4 on its PDF.
- Answers can still contain errors — the UI shows a disclaimer and always cites sources so claims are verifiable.
- The reranker and judge are LLMs, so scores carry mild, expected variance.

### Troubleshooting

- **Port already in use** — run on another port: `uvicorn api.main:app --port 8600`.
- **`tiktoken` can't download its encoding (offline/restricted network)** — pre-seed its cache and set
  `export TIKTOKEN_CACHE_DIR=/path/to/cache`. Only needed for the ingest stages, not for serving.

---

## 🗺️ Roadmap

- Stream the answer token-by-token (feels faster)
- Multi-document support (pick a company, filter by metadata)
- Deploy behind a public URL

---

<p align="center"><sub>Built as an end-to-end RAG learning project — parse → chunk → embed → store → retrieve → rerank → generate → evaluate.</sub></p>
