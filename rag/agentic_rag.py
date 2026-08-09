"""
rag/agentic_rag.py

Stage 4/5 -- Agentic GraphRAG: LangGraph orchestrator over vector + graph retrieval.

Pipeline (conditional DAG):
  decomposer -> router -> [execution_mode == "parallel"] -> vector_node -> graph_node -> synthesiser
                       -> [execution_mode == "serial"  ] -> hop_executor               -> synthesiser

  decomposer   : breaks question into <= max_sub_questions atomic sub-questions.
                 Stage 5: emits execution_mode, cypher_sufficient, hop_dependencies.
  router       : classifies each sub-question as "vector" | "graph" | "both"
  vector_node  : Stage 4 parallel path — ChromaDB + entity pre-filter + cross-encoder rerank
  graph_node   : Stage 4 parallel path — NL→Cypher→Neo4j per sub-question
  hop_executor : Stage 5 serial path — dependency-aware execution with template resolution
  synthesiser  : merges both sources and generates the final answer with inline citations

Stage 5 routing decision (made in _route_after_router()):
  execution_mode == "parallel" → existing Stage 4 path (vector_node → graph_node)
  execution_mode == "serial"   → hop_executor (handles both cypher_sufficient paths)

Fallback chain (per CLAUDE.md s6):
  graph_node / hop_executor fails → synthesiser uses vector_chunks only
  both fail                       → "I don't have enough information"

Public API:
    AgenticResult              -- TypedDict returned by answer()
    answer(question, cfg)      -- full Stage 4/5 pipeline, never raises
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
from rag.nodes.retrieval import graph_node, vector_node
from rag.nodes.router import route
from rag.nodes.synthesiser import synthesise

load_dotenv()

log = logging.getLogger(__name__)


# -- Return type --------------------------------------------------------------

class AgenticResult(TypedDict):
    question:      str        # original question (for RAGAS)
    answer:        str        # synthesised answer with citations
    contexts:      list[str]  # vector chunks + graph row strings (RAGAS context metrics)
    graph_rows:    list[dict] # flattened Neo4j rows (node_precision / cypher_validity)
    sub_questions: list[str]  # decomposed sub-questions (routing_accuracy logging)
    routes:        list[str]  # routing decisions per sub-question
    stage:         str        # always "agentic"
    model:         str        # system model name from config
    # Stage 5 fields — populated for all queries (defaults for Stage 4 parallel path)
    execution_mode:    str    # "parallel" | "serial"
    cypher_sufficient: bool   # True → single-shot Cypher was used
    hop_dependencies:  list   # per-hop dependency list
    hop_results:       dict   # hop_id -> resolved entity name


# -- Graph row → context string -----------------------------------------------

def _graph_rows_to_context_strings(graph_results: list) -> list[str]:
    """
    Convert successful Neo4j graph result rows into plain text strings
    suitable for RAGAS context evaluation.

    Uses the same key=value formatting as synthesiser._format_row() so the
    context strings match what the synthesiser actually reasons over.
    Deduplicates across all graph results (same row can appear from multiple
    sub-questions resolved by the same Cypher path).

    Returns an empty list when no graph results succeeded.
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


# -- LangGraph graph assembly -------------------------------------------------

_app = None  # compiled graph singleton -- rebuilt for Stage 5 (6 nodes, conditional DAG)


def _route_after_router(state: FinGraphState) -> str:
    """
    Conditional edge function called after the router node.

    Returns the name of the next node to execute:
      "parallel" → vector_node (Stage 4 unchanged path)
      "serial"   → hop_executor (Stage 5 dependency-aware path)
    """
    mode = state.get("execution_mode", "parallel")
    return "hop_executor" if mode == "serial" else "vector_node"


def _get_app():
    global _app
    if _app is None:
        g = StateGraph(FinGraphState)
        g.add_node("decomposer",   decompose)
        g.add_node("router",       route)
        g.add_node("vector_node",  vector_node)
        g.add_node("graph_node",   graph_node)
        g.add_node("hop_executor", hop_execute)
        g.add_node("synthesiser",  synthesise)

        g.set_entry_point("decomposer")
        g.add_edge("decomposer", "router")

        # Conditional branch: parallel (Stage 4) vs serial (Stage 5)
        g.add_conditional_edges(
            "router",
            _route_after_router,
            {
                "vector_node":  "vector_node",   # parallel path
                "hop_executor": "hop_executor",  # serial path
            },
        )

        # Parallel path edges (Stage 4 unchanged)
        g.add_edge("vector_node", "graph_node")
        g.add_edge("graph_node",  "synthesiser")

        # Serial path edge (Stage 5)
        g.add_edge("hop_executor", "synthesiser")

        g.add_edge("synthesiser", END)

        _app = g.compile()
        log.info(
            "LangGraph Stage 5 compiled (6 nodes, conditional DAG: "
            "parallel→vector_node+graph_node | serial→hop_executor)"
        )
    return _app


# -- Main pipeline ------------------------------------------------------------

def answer(question: str, cfg: dict | None = None) -> AgenticResult:
    """
    Stage 3 Agentic GraphRAG pipeline.

    Invokes the compiled LangGraph DAG and returns a structured result
    compatible with the RAGAS eval pipeline and MLflow logging.
    Never raises -- node failures degrade gracefully within the DAG.
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
        # Stage 5 defaults — decomposer will overwrite if multi-hop is detected
        "execution_mode":    "parallel",
        "cypher_sufficient": False,
        "hop_dependencies":  [],
        "hop_results":       {},
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
        )
    invoke_latency = time.perf_counter() - _t

    # Flatten graph rows for MLflow / RAGAS logging
    all_rows: list[dict] = []
    for gr in final.get("graph_results", []):
        all_rows.extend(gr["rows"])

    # Build unified context list: vector chunks + graph row strings.
    # Graph rows are retrieved and passed to the synthesiser but were previously
    # excluded from AgenticResult.contexts (RAGAS never saw them).
    # Including them fixes context_entity_recall -- graph rows contain canonical
    # entity names and factual fields (penalty_lakh, date, violation text) that
    # RAGAS checks against ground truth answers.
    graph_ctx = _graph_rows_to_context_strings(final.get("graph_results", []))
    all_contexts: list[str] = final.get("vector_chunks", []) + graph_ctx

    routes: list[str] = final.get("routes", [])
    logger.step(
        "langgraph_invoke",
        latency_s           = invoke_latency,
        n_sub_questions     = len(final.get("sub_questions", [])),
        n_vector_chunks     = len(final.get("vector_chunks", [])),
        n_graph_ctx_strings = len(graph_ctx),
        n_total_contexts    = len(all_contexts),
        n_graph_rows        = len(all_rows),
        n_graph_results     = len(final.get("graph_results", [])),
        graph_success_count = sum(
            1 for gr in final.get("graph_results", []) if gr.get("success")
        ),
        routes_graph  = routes.count("graph"),
        routes_vector = routes.count("vector"),
        routes_both   = routes.count("both"),
    )

    ans = final["answer"]
    logger.finish(
        ans,
        n_contexts        = len(all_contexts),
        n_vector_contexts = len(final.get("vector_chunks", [])),
        n_graph_contexts  = len(graph_ctx),
        n_graph_rows      = len(all_rows),
    )

    return AgenticResult(
        question=question,
        answer=ans,
        contexts=all_contexts,
        graph_rows=all_rows,
        sub_questions=final["sub_questions"],
        routes=routes,
        stage="agentic",
        model=model,
        execution_mode=final.get("execution_mode", "parallel"),
        cypher_sufficient=final.get("cypher_sufficient", False),
        hop_dependencies=final.get("hop_dependencies", []),
        hop_results=final.get("hop_results", {}),
    )


# -- Smoke test ---------------------------------------------------------------

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
        print(f"   Chunks : {len(_r['contexts'])} vector | {len(_r['graph_rows'])} graph rows")
        print(f"   A      : {_r['answer']}")
        print(f"   stage={_r['stage']}")
