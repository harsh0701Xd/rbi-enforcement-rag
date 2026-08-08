# FinGraph Intelligence System
### Multi-Hop Agentic RAG over RBI Enforcement Data

A production-grade Retrieval-Augmented Generation system that answers natural language queries against a corpus of **~650 RBI enforcement press releases** (2021–2025). The system combines a **ChromaDB vector store** with a **Neo4j knowledge graph**, evaluated rigorously across five RAG architectures using RAGAS.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LangGraph Agentic Pipeline (Stage 4)                   │
│                                                         │
│  Decomposer → Router → ┌─ Vector Node ─┐               │
│                         │  ChromaDB     │ → Synthesiser │
│                         │  + Reranker   │               │
│                         ├─ Graph Node ──┤               │
│                         │  Neo4j Cypher │               │
│                         └───────────────┘               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Grounded Answer (with PR citations)
```

**Key components:**

- **Ingestion pipeline** — scrapes RBI press releases, extracts structured entities (bank name, penalty amount, violation type, statutory citations) with Claude Haiku, writes to both ChromaDB (1,300 chunks, nomic-embed-text 768-dim) and Neo4j (entity/penalty/violation graph)
- **Stage 4 Agentic RAG** — LangGraph graph with query decomposition, hybrid routing, entity pre-filtering, cross-encoder reranking (ms-marco), and graph Cypher generation
- **RAGAS evaluation** — 6-metric evaluation (faithfulness, answer relevancy, context precision/recall, answer correctness, context entity recall) across all 5 stages using Claude Sonnet as judge

---

## RAG Progression & RAGAS Results

| Stage | System | Composite | Faithfulness | Ans. Relevancy | Ctx. Precision | Ctx. Recall | Ans. Correctness |
|---|---|---|---|---|---|---|---|
| 1 | Naive RAG | 0.491 | **0.829** | 0.652 | 0.421 | 0.327 | 0.325 |
| 2 | Advanced v1 | 0.479 | 0.691 | 0.644 | 0.450 | 0.417 | 0.335 |
| 2 | Advanced v2 | 0.558 | 0.738 | 0.787 | 0.513 | 0.437 | 0.482 |
| 3 | Agentic v1 | 0.574 | 0.646 | **0.907** | 0.509 | 0.640 | 0.449 |
| 4 | **Agentic v2** | **0.589** | 0.697 | 0.880 | **0.575** | **0.643** | 0.414 |

> Composite = mean of all 6 RAGAS metrics. Judge model: `claude-sonnet-4-6`. Eval set: 25 QA pairs across 3 tiers (factual, relational, analytical).

**Stage 4 key upgrades over Stage 3:**
- Entity name pre-filter (`$in` operator) narrows ChromaDB to the named institution before cosine search
- Wide retrieval (top-k=10) → cross-encoder reranking (top-k_final=5) per sub-question
- Graph rows included in synthesiser context alongside vector chunks

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude Haiku 4.5 (system), Claude Sonnet 4.6 (judge) |
| Orchestration | LangGraph 0.1.19 |
| Vector store | ChromaDB 0.5.5 + nomic-embed-text (768-dim, via Ollama) |
| Graph store | Neo4j 5 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (in-process) |
| Evaluation | RAGAS 0.1.14 + MLflow 2.14 |
| Serving | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Infra | Docker Compose, EC2 (prod) |

---

## Project Structure

```
rbi_intel/
├── ingestion/          # Scraper, extractor, graph writer, vector writer
├── retrieval/          # vector_retriever.py, graph_retriever.py (NL→Cypher)
├── rag/
│   ├── nodes/          # Decomposer, router, vector_node, graph_node, synthesiser
│   ├── naive_rag.py    # Stage 1
│   ├── advanced_rag.py # Stage 2 (HyDE + reranking)
│   └── agentic_rag.py  # Stage 3/4 (LangGraph)
├── prompts/            # Cypher, decomposer, synthesis, HyDE, extraction prompts
├── eval/
│   ├── ground_truth.json      # 25 locked QA pairs (3 tiers)
│   ├── ragas_eval.py          # Evaluation runner
│   └── results/               # RAGAS JSON outputs for all stages
├── tests/
│   ├── unit/           # 6 unit test modules
│   └── integration/    # Stage 4 live integration tests (15 smoke tests)
├── patches/            # ChromaDB 0.5.5 bug fixes (see below)
├── deploy/
│   ├── docker-compose.yml
│   └── .github/workflows/
├── query_cli.py        # Interactive pre-deployment validation CLI
├── config.yaml         # All hyperparameters (top-k, temperatures, models)
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11
- Docker + Docker Compose
- [Ollama](https://ollama.ai) with `nomic-embed-text` pulled (`ollama pull nomic-embed-text`)
- Anthropic API key
- Neo4j 5 (Docker or [Aura Free](https://neo4j.com/cloud/platform/aura-graph-database/))

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rbi-intel.git
cd rbi-intel/rbi_intel
python3.11 -m venv venv
venv/bin/python3 -m pip install -r requirements.txt
```

### 2. Apply ChromaDB patches

ChromaDB 0.5.5 has two bugs when reading stores created by older versions. Apply the patches after install:

```bash
bash patches/apply_patches.sh
```

See [`patches/`](patches/) for the detailed diffs and root-cause explanation.

### 3. Configure environment

```bash
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, NEO4J_URI, NEO4J_PASSWORD, PG_USER, PG_PASSWORD
```

### 4. Start infrastructure

```bash
docker compose -f deploy/docker-compose.yml --env-file .env up neo4j postgres mlflow -d
```

### 5. Run ingestion

```bash
# Scrape RBI press releases
venv/bin/python3 -m ingestion.rbi_enforcement_scraper

# Extract entities + write to Neo4j and ChromaDB
venv/bin/python3 -m ingestion.extractor
venv/bin/python3 -m ingestion.graph_writer
venv/bin/python3 -m ingestion.vector_writer
```

### 6. Run the interactive query CLI

```bash
venv/bin/python3 query_cli.py
```

The CLI shows full pipeline internals per query: decomposition, routing, entity pre-filter, vector chunks, graph rows, and final answer.

---

## Evaluation

```bash
# Run RAGAS eval for a specific stage
venv/bin/python3 eval/ragas_eval.py --stage agentic

# Run with suffix to avoid overwriting baselines
venv/bin/python3 eval/ragas_eval.py --stage agentic --suffix v2

# Dry run (cheap validation, no judge LLM cost)
venv/bin/python3 eval/dry_run.py
```

Results are saved to `eval/results/<stage>_<tier>.json` and logged to MLflow.

---

## Testing

```bash
# Unit tests (no external services required)
venv/bin/python3 -m pytest tests/unit/ -v

# Integration tests (requires running Neo4j + ChromaDB)
venv/bin/python3 -m pytest tests/integration/ -v
```

---

## ChromaDB Patches

Two bugs in ChromaDB 0.5.5 surface when reading a store created by an older ingestion run:

**Bug 1 — `sqlite.py: _decode_seq_id`**
SQLite stores `seq_id` as `INTEGER` (older versions) but ChromaDB 0.5.5 calls `len()` on it expecting `bytes`, raising `TypeError: object of type 'int' has no len()`.
Fix: add `isinstance(seq_id_bytes, int)` guard to return the int directly.

**Bug 2 — `local_persistent_hnsw.py: PersistentData.load_from_file`**
Older ingestion serialised HNSW index metadata as a plain `dict`. ChromaDB 0.5.5 calls `.dimensionality` on the loaded object, raising `AttributeError: 'dict' object has no attribute 'dimensionality'`.
Fix: detect `dict` on load and wrap into a `PersistentData` instance.

Both patches are in [`patches/`](patches/) with a one-line apply script.

---

## Known Limitations & Roadmap

**Stage 5 (planned):** Dependency-aware multi-hop execution.

The current Stage 4 pipeline decomposes queries and runs all retrievals in parallel. For queries where sub-question 2 depends on sub-question 1's answer (e.g. "For the bank with the highest penalty in 2023, what were the specific violations?"), the entity pre-filter on sub-question 2 cannot use sub-question 1's result because retrieval is parallel.

Stage 5 will add:
- **`cypher_sufficient` routing flag** — fully graph-answerable multi-hop queries generate a single `WITH`-chained Cypher query (one DB round-trip, no serial latency)
- **Dependency-aware serial execution** — hybrid queries (graph for hop 1, vector for hop 2) run sequentially with template variable substitution (`{q1_answer}`) between hops, ensuring the entity filter activates correctly on dependent sub-questions
- **`max_hops = 3` guard** — prevents latency compounding and error propagation on deep chains

---

## License

MIT
