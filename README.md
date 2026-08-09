# FinGraph Intelligence System
### Multi-Hop Agentic RAG over RBI Enforcement Data

A production-grade Retrieval-Augmented Generation system that answers natural language queries against a corpus of **~650 RBI enforcement press releases** (2021–2025). The system combines a **ChromaDB vector store** with a **Neo4j knowledge graph**, evaluated rigorously across five RAG architectures using RAGAS.

---

## Architecture Overview

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  LangGraph Agentic Pipeline (Stage 5)                            │
│                                                                  │
│  Decomposer ──► Router ──► [execution_mode == parallel?]         │
│                                    │                             │
│                    ┌───── Yes ──────┴────── No ─────┐            │
│                    ▼                                ▼            │
│            ┌─ Vector Node ─┐               Hop Executor          │
│            │  ChromaDB     │               (serial hops)         │
│            │  + Reranker   │  → Synthesiser     │                │
│            ├─ Graph Node ──┤       ▲             │               │
│            │  Neo4j Cypher │       └─────────────┘               │
│            └───────────────┘                                     │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
Grounded Answer (with PR citations)
```

**Key components:**

- **Ingestion pipeline** — scrapes RBI press releases, extracts structured entities (bank name, penalty amount, violation type, statutory citations) with Claude Haiku, writes to both ChromaDB (1,300 chunks, nomic-embed-text 768-dim) and Neo4j (entity/penalty/violation graph)
- **Stage 5 Agentic RAG** — LangGraph conditional DAG with dependency-aware multi-hop execution, query decomposition, hybrid routing, entity pre-filtering, cross-encoder reranking, and graph Cypher generation
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
| 5 | **Agentic v3** | — | — | — | — | — | — |

> Composite = mean of all 6 RAGAS metrics. Judge model: `claude-sonnet-4-6`. Eval set: 25 QA pairs across 3 tiers (factual, relational, analytical).
> Stage 5 RAGAS scores are not directly comparable to Stages 1–4 on the same 25-question set — Stage 5 addresses a categorical capability gap (dependent multi-hop reasoning) rather than an incremental metric improvement on existing query types.

**Stage 4 key upgrades over Stage 3:**
- Entity name pre-filter (`$in` operator) narrows ChromaDB to the named institution before cosine search
- Wide retrieval (top-k=10) → cross-encoder reranking (top-k_final=5) per sub-question
- Graph rows included in synthesiser context alongside vector chunks

**Stage 5 key upgrades over Stage 4:**
- Decomposer emits JSON with `execution_mode`, `cypher_sufficient`, and per-hop `depends_on` fields
- `cypher_sufficient=True` path — fully graph-answerable multi-hop queries generate a single `WITH`-chained Cypher (one DB round-trip, no serial latency overhead)
- `cypher_sufficient=False` hybrid path — graph identifies the entity on hop 0, template resolver substitutes `{q0_answer}` into hop 1's sub-question before `extract_entities()` runs, so the ChromaDB entity pre-filter activates correctly on dependent hops
- `max_hops=3` guard prevents latency compounding on deep chains

---

## Stage 5: Multi-Hop Dependent Reasoning

Stage 4 ran all sub-questions in parallel. This worked for independent queries but failed for dependent ones — if sub-question 2 was "What violations did {that bank} commit?", the entity pre-filter on sub-question 2 couldn't use sub-question 1's answer because retrieval was already in flight.

Stage 5 fixes this with a conditional LangGraph DAG:

**Example — hybrid path (`cypher_sufficient=False`):**
```
Q: "Which cooperative bank was penalised the most times, and what did the press release say?"

Hop 0 (graph):  COUNT(PenaltyAction) GROUP BY Institution → "Bharat Co-operative Bank"
                hop_results[0] = "Bharat Co-operative Bank (Mumbai) Limited"

Hop 1 (vector): resolve "{q0_answer}" → "Bharat Co-operative Bank (Mumbai) Limited"
                extract_entities(resolved_q) → entity filter activates
                ChromaDB retrieves only Bharat Co-operative Bank press releases ✓
```

**Example — graph-only path (`cypher_sufficient=True`):**
```
Q: "Which bank had the highest penalty in 2023 and what were the violations?"

Single WITH-chained Cypher:
  MATCH (i:Institution)-[:RECEIVED]->(p:PenaltyAction)
  WHERE p.date >= "2023-01-01" AND p.date < "2024-01-01"
  WITH i, p ORDER BY p.penalty_lakh DESC LIMIT 1
  OPTIONAL MATCH (p)-[:FOR_VIOLATION]->(v:Violation)
  RETURN i.name, p.penalty_lakh, p.date, v.text   → 20 rows, 1 DB round-trip ✓
```

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
├── retrieval/          # vector_retriever.py, graph_retriever.py (NL→Cypher + multi-hop)
├── rag/
│   ├── nodes/          # decomposer, router, vector_node, graph_node,
│   │                   # hop_executor (Stage 5), synthesiser
│   ├── _entities.py    # Named entity registry + ChromaDB filter helpers
│   ├── _resolver.py    # Template variable resolution ({q0_answer} substitution)
│   ├── _reranker.py    # Cross-encoder reranking (ms-marco)
│   ├── _state.py       # FinGraphState TypedDict (shared across all nodes)
│   ├── naive_rag.py    # Stage 1
│   ├── advanced_rag.py # Stage 2 (HyDE + reranking)
│   └── agentic_rag.py  # Stage 3–5 (LangGraph conditional DAG)
├── prompts/            # Cypher, multi-hop Cypher, decomposer, synthesis prompts
├── eval/
│   ├── ground_truth.json           # 25 locked QA pairs (3 tiers, Stages 1–4)
│   ├── ground_truth_multihop.json  # 5 multi-hop QA pairs (Stage 5 validation)
│   ├── ragas_eval.py               # Evaluation runner
│   └── results/                    # RAGAS JSON outputs for all stages
├── tests/
│   ├── unit/           # 6 unit test modules
│   └── integration/    # Stage 5 live integration tests
├── patches/            # ChromaDB 0.5.5 bug fixes (see below)
├── deploy/
│   ├── docker-compose.yml
│   └── .github/workflows/
├── query_cli.py        # Interactive validation CLI (--debug flag for pipeline logs)
├── config.yaml         # All hyperparameters (top-k, temperatures, max_hops, models)
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
venv/bin/python3 query_cli.py            # clean output
venv/bin/python3 query_cli.py --debug    # show pipeline logs (routing, entity filter, hop resolution)
```

The CLI shows full pipeline internals per query: execution mode (parallel/serial), decomposition, routing, entity pre-filter, vector chunks, graph rows, hop resolution, and final answer.

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

## License

MIT
