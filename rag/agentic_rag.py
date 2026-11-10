"""
rag/agentic_rag.py

Stage 6 — Agentic GraphRAG: simplified 3-node LangGraph DAG.

Pipeline (fixed linear DAG, no conditional edges):
  decomposer → hop_executor → synthesiser

  decomposer   : breaks question into atomic sub-questions; emits cypher_sufficient,
                 hop_dependencies, fan_out.
  hop_executor : owns ALL retrieval — handles three internal paths without DAG branching:
                   Path A (cypher_sufficient=True) → single-shot WITH-chained Cypher
                   Path B (fan_out=True)            → iterate Q2 over all Q1 rows
                   Path C (default)                 → hybrid serial, two-pass
  synthesiser  : merges context and generates final answer with inline citations.

Stage 6 removals vs Stage 5:
  - vector_node, graph_node removed from DAG (hop_executor handles all retrieval)
  - _route_after_decomposer conditional edge removed (no parallel path)
  - execution_mode no longer drives routing (always "serial" — kept for UI compat only)
  - fan_out added to state, AgenticResult, and API response

Fallback chain (per CLAUDE.md §6):
  graph_node / hop_executor Cypher fails → synthesiser uses vector_chunks only
  both fail                             → "I don't have enough information"

Public API:
    AgenticResult              -- TypedDict returned by answer()
    answer(question, cfg)      -- Stage 6 pipeline, never raises
"""

from __future__ import annotations

import logging
import time
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from config import get_config
from rag._logger import InferenceLogger
from rag._state import FinGraphState
from rag.nodes.decomposer import decompose
from rag.nodes.hop_executor import hop_execute
from rag.nodes.synthesiser import synthesise

load_dotenv()

log = logging.getLogger(__name__)


# ── Return type ───────────────────────────────────────────────────────────────

class AgenticResult(TypedDict):
    question:      str        # original question (for RAGAS)
    answer:        str        # synthesised answer with citations
    contexts:      list[str]  # vector chunks + graph row strings (RAGAS context metrics)
    graph_rows:    list[dict] # flattened Neo4j rows (node_precision / cypher_validity)
    sub_questions: list[str]  # decomposed sub-questions (routing_accuracy logging)
    routes:        list[str]  # routing decisions per sub-question
    stage:         str        # always "agentic"
    model:         str        # system model name from config
    # Execution metadata
    execution_mode:    str        # always "serial" in Stage 6 (kept for UI backward-compat)
    cypher_sufficient: bool       # True → single-shot WITH-chained Cypher was used
    fan_out:           bool       # True → Q2 iterated over all Q1 result rows
    hop_dependencies:  list       # per-hop dependency list
    hop_results:       dict       # hop_id -> raw rows[0] dict (Stage 6)
    cypher_queries:    list[dict] # {label, cypher, success, n_rows} per graph attempt
    n_graph_attempted: int        # total graph retrieve calls made
    n_graph_succeeded: int        # how many returned success=True


# ── Graph row → context string ─────────────────────────────────────────────────

def _graph_rows_to_context_strings(graph_results: list) -> list[str]:
    """
    Convert successful Neo4j graph result rows into plain text context strings
    suitable for RAGAS evaluation.

    Uses the same key=value formatting as synthesiser._format_row() so the
    context strings match what the synthesiser actually reasons over.
    Deduplicates across all graph results.
    """
    strings: list[str] = []
    seen:    set[str]  = set()

    for gr in graph_results:
        if not gr.get("success") or not gr.get("rows"):
            continue
        for row in gr["rows"]:
            parts = [
                f"{k.split('.')[-1]}: {v}"
                for k, v in row.items()
                if v is not None and str(v).strip()
            ]
            if not parts:
                continue
            s = " | ".join(parts)
            if s not in seen:
                seen.add(s)
                strings.append(s)

    return strings


# ── LangGraph graph assembly ──────────────────────────────────────────────────

_app = None  # compiled graph singleton


def _get_app():
    global _app
    if _app is None:
        g = StateGraph(FinGraphState)

        # Stage 6: 3-node linear DAG — no conditional edges, no parallel path
        g.add_node("decomposer",   decompose)
        g.add_node("hop_executor", hop_execute)
        g.add_node("synthesiser",  synthesise)

        g.set_entry_point("decomposer")
        g.add_edge("decomposer",   "hop_executor")
        g.add_edge("hop_executor", "synthesiser")
        g.add_edge("synthesiser",  END)

        _app = g.compile()
        log.info(
            "LangGraph compiled (Stage 6: 3-node linear DAG — "
            "decomposer → hop_executor → synthesiser, no conditional edges)"
        )
    return _app


# ── Main pipeline ─────────────────────────────────────────────────────────────

def answer(question: str, cfg: dict | None = None) -> AgenticResult:
    """
    Stage 6 Agentic GraphRAG pipeline.

    Invokes the compiled LangGraph DAG and returns a structured result
    compatible with the RAGAS eval pipeline and MLflow logging.
    Never raises — node failures degrade gracefully within the DAG.
    """
    if cfg is None:
        cfg = get_config()

    model: str = cfg["models"]["system_model"]
    app    = _get_app()
    logger = InferenceLogger("agentic", question)

    initial: FinGraphState = {
        "question":      question,
        "sub_questions": [],
        "routes":        [],
        "vector_chunks": [],
        "graph_results": [],
        "context":       "",
        "answer":        "",
        "iterations":    0,
        # Stage 5/6 defaults — decomposer will overwrite
        "execution_mode":    "serial",   # always serial in Stage 6
        "cypher_sufficient": False,
        "hop_dependencies":  [],
        "hop_results":       {},
        # Stage 6 default
        "fan_out": False,
    }

    _t = time.perf_counter()
    try:
        final = app.invoke(initial)
    except Exception as exc:
        log.error("LangGraph invocation failed: %s", exc)
        ans = "An error occurred in the agentic pipeline. Please try again."
        logger.step("langgraph_invoke", latency_s=time.perf_counter() - _t, error=True)
        logger.finish(ans, n_contexts=0)
        return AgenticResult(
            question=question, answer=ans,
            contexts=[], graph_rows=[], sub_questions=[question],
            routes=["graph"], stage="agentic", model=model,
            execution_mode="serial", cypher_sufficient=False, fan_out=False,
            hop_dependencies=[], hop_results={},
            cypher_queries=[], n_graph_attempted=0, n_graph_succeeded=0,
        )
    invoke_latency = time.perf_counter() - _t

    # Flatten graph rows for MLflow / RAGAS logging
    all_rows: list[dict] = []
    _grs_all = final.get("graph_results", [])
    for gr in _grs_all:
        all_rows.extend(gr["rows"])

    # Build unified context: vector chunks + graph row strings
    graph_ctx     = _graph_rows_to_context_strings(_grs_all)
    all_contexts  = final.get("vector_chunks", []) + graph_ctx
    routes: list[str] = final.get("routes", [])

    # Derived metrics for MLflow
    _n_zero_row     = sum(1 for gr in _grs_all if gr.get("success") and not gr["rows"])
    _cypher_in      = sum(gr.get("input_tokens",  0) for gr in _grs_all)
    _cypher_out     = sum(gr.get("output_tokens", 0) for gr in _grs_all)
    _cypher_suff    = final.get("cypher_sufficient", False)
    _fan_out_flag   = final.get("fan_out", False)
    _mode           = "single_shot" if _cypher_suff else ("fan_out" if _fan_out_flag else "serial")

    logger.step(
        "langgraph_invoke",
        latency_s           = invoke_latency,
        n_sub_questions     = len(final.get("sub_questions", [])),
        n_vector_chunks     = len(final.get("vector_chunks", [])),
        n_graph_rows        = len(all_rows),
        n_graph_results     = len(_grs_all),
        graph_success_count = sum(1 for gr in _grs_all if gr.get("success")),
        n_zero_row_successes= _n_zero_row,
        cypher_sufficient   = int(_cypher_suff),
        fan_out             = int(_fan_out_flag),
        cypher_token_input  = _cypher_in,
        cypher_token_output = _cypher_out,
        routes_graph        = routes.count("graph"),
        routes_vector       = routes.count("vector"),
        routes_both         = routes.count("both"),
    )

    ans = final["answer"]
    logger.finish(
        ans,
        n_contexts = len(all_contexts),
        mode       = _mode,
    )

    # Build per-hop Cypher metadata for UI display.
    # Includes ALL attempts (success + failure) so the UI can show X/Y succeeded.
    _sub_qs      = final.get("sub_questions", [])
    _n_sub       = len(_sub_qs)
    _grs_with_cq = [gr for gr in _grs_all if gr.get("cypher")]

    cypher_queries: list[dict] = []
    for gr in _grs_with_cq:
        sub_idx = gr.get("sub_q_idx", 0)
        # Fan-out + graph path collapses to retrieve_multihop (single Cypher).
        # Label it clearly so the user knows it covers all sub-questions,
        # not just Q1. (Same treatment as cypher_sufficient=True.)
        if _fan_out_flag and len(_grs_with_cq) == 1 and _n_sub > 1:
            covered = ", ".join(f"Q{j+1}" for j in range(_n_sub))
            label   = f"Fan-out multi-hop ({covered})"
        else:
            label = f"Q{sub_idx + 1}"
        cypher_queries.append({
            "label":   label,
            "cypher":  gr.get("cypher", ""),
            "success": bool(gr.get("success")),
            "n_rows":  len(gr.get("rows", [])),
        })
    n_graph_attempted: int = len(_grs_all)
    n_graph_succeeded: int = sum(1 for gr in _grs_all if gr.get("success"))

    return AgenticResult(
        question=question,
        answer=ans,
        contexts=all_contexts,
        graph_rows=all_rows,
        sub_questions=final["sub_questions"],
        routes=routes,
        stage="agentic",
        model=model,
        execution_mode=final.get("execution_mode", "serial"),
        cypher_sufficient=final.get("cypher_sufficient", False),
        fan_out=final.get("fan_out", False),
        hop_dependencies=final.get("hop_dependencies", []),
        hop_results=final.get("hop_results", {}),
        cypher_queries=cypher_queries,
        n_graph_attempted=n_graph_attempted,
        n_graph_succeeded=n_graph_succeeded,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    _cfg = get_config()
    _tests = [
        "What penalty did RBI impose on Axis Bank and for what reason?",
        "Which NBFCs had their Certificate of Registration cancelled by RBI?",
        "Which banks were penalised for KYC non-compliance in 2023?",
    ]
    for _q in _tests:
        _r = answer(_q, cfg=_cfg)
        print(f"\nQ: {_q}")
        print(f"   Sub-Qs : {_r['sub_questions']}")
        print(f"   Routes : {_r['routes']}")
        print(f"   fan_out: {_r['fan_out']}")
        print(f"   Chunks : {len(_r['contexts'])} contexts | {len(_r['graph_rows'])} graph rows")
        print(f"   A      : {_r['answer'][:200]}")
