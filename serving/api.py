"""
serving/api.py

FastAPI serving layer for FinGraph Intelligence System.

Endpoints:
  GET  /health   — liveness check (used by Docker healthcheck and load-balancer)
  POST /query    — run a question through the Stage 5 agentic pipeline

The response includes all pipeline metadata so the Streamlit frontend can
render execution mode, hop resolution, decomposition, and retrieval counts
without a second round-trip.

Run locally (from rbi_intel/):
  uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload

In Docker:
  CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on sys.path when uvicorn is started from outside the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_config
from rag.agentic_rag import answer as _run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# Load config once at process startup — shared across all requests
_cfg = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm expensive models at startup so the first query doesn't pay cold-start cost."""
    log.info("Startup: pre-warming cross-encoder reranker...")
    try:
        from rag._reranker import _get_reranker
        _get_reranker(_cfg)
        log.info("Startup: cross-encoder reranker ready.")
    except Exception as exc:
        log.warning("Startup: reranker pre-warm failed (non-fatal): %s", exc)
    yield
    # shutdown — nothing to clean up


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    lifespan=lifespan,
    title="FinGraph Intelligence System",
    description="Multi-hop agentic RAG over RBI enforcement data (Stage 5)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to Streamlit origin in prod
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question about RBI enforcement data",
        examples=["Which bank received the highest penalty in 2023?"],
    )


class QueryResponse(BaseModel):
    question: str
    answer:   str

    # ── Routing & decomposition ──────────────────────────────────────────────
    execution_mode:    str        # always "serial" in Stage 6 (kept for UI compat)
    cypher_sufficient: bool       # True → single-shot WITH-chained Cypher used
    fan_out:           bool       # True → Q2 iterated over all Q1 result rows
    sub_questions:     list[str]
    routes:            list[str]  # "vector" | "graph" | "both" per sub-question
    hop_dependencies:  list       # None or int per hop
    hop_results:       dict       # hop_id (str) → raw rows[0] dict (Stage 6)

    # ── Retrieval detail (for Streamlit full-detail view) ────────────────────
    contexts:          list[str]   # vector chunks + graph ctx strings
    graph_rows:        list[dict]  # raw Neo4j result rows
    cypher_queries:    list        # [{label, cypher, success, n_rows}] per graph attempt
    n_vector_chunks:   int
    n_graph_rows:      int
    n_graph_attempted: int
    n_graph_succeeded: int

    # ── Timing & metadata ────────────────────────────────────────────────────
    latency_s: float
    stage:     str    # always "agentic"
    model:     str


class HealthResponse(BaseModel):
    status: str
    stage:  str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — returns 200 when the process is alive."""
    return HealthResponse(status="ok", stage="agentic_v5")


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest) -> QueryResponse:
    """
    Run a question through the Stage 5 agentic RAG pipeline.

    The response includes full pipeline metadata (execution mode, hop results,
    routing decisions, retrieval counts, and context strings) so the frontend
    can render pipeline internals without a second round-trip.

    Execution modes:
    - **parallel** (Stage 4 path): all sub-questions retrieved concurrently
    - **serial** (Stage 5 path): dependent hops resolved sequentially with
      template variable substitution (`{q0_answer}`)
    """
    log.info("Query received | %s", req.question[:120])
    t0 = time.perf_counter()

    try:
        result = _run_pipeline(req.question, cfg=_cfg)
    except Exception as exc:
        log.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    latency = round(time.perf_counter() - t0, 3)
    log.info(
        "Query complete | latency=%.2fs | cypher_ok=%s | fan_out=%s | chunks=%d | rows=%d",
        latency,
        result["cypher_sufficient"],
        result["fan_out"],
        len(result["contexts"]),
        len(result["graph_rows"]),
    )

    # Separate vector chunks from graph context strings for the frontend.
    # Vector chunks are written by the ingestion pipeline with an "Entity:" prefix;
    # graph context strings are formatted "key: val | key: val" by agentic_rag.py.
    vector_chunks = [c for c in result["contexts"] if c.startswith("Entity:")]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        execution_mode=result["execution_mode"],
        cypher_sufficient=result["cypher_sufficient"],
        fan_out=result.get("fan_out", False),
        sub_questions=result["sub_questions"],
        routes=result["routes"],
        hop_dependencies=result["hop_dependencies"],
        # JSON only allows string keys — cast int hop_ids to str
        # Values are now raw rows[0] dicts (Stage 6), not strings
        hop_results={str(k): v for k, v in result["hop_results"].items()},
        contexts=result["contexts"],
        graph_rows=result["graph_rows"],
        cypher_queries=result.get("cypher_queries", []),
        n_vector_chunks=len(vector_chunks),
        n_graph_rows=len(result["graph_rows"]),
        n_graph_attempted=result.get("n_graph_attempted", 0),
        n_graph_succeeded=result.get("n_graph_succeeded", 0),
        latency_s=latency,
        stage=result["stage"],
        model=result["model"],
    )
