"""
rag/nodes/hop_executor.py

LangGraph node: serial, dependency-aware multi-hop executor (Stage 5).

Called instead of vector_node + graph_node when execution_mode == "serial".
Implements two execution paths:

  cypher_sufficient=True  (graph-only multi-hop)
  ─────────────────────────────────────────────
  The decomposer determined all sub-questions can be answered from Neo4j.
  Calls retrieve_multihop() to generate and execute a single WITH-chained
  Cypher in one DB round-trip.  Falls back to hybrid if Cypher generation
  fails.

  cypher_sufficient=False  (hybrid serial)
  ─────────────────────────────────────────
  Runs hops in two passes:

    Pass 1 — independent hops (depends_on=None):
      Graph/vector retrieval as normal.
      If the hop has a graph result, the first row's primary entity name is
      stored in hop_results[hop_id] for template substitution.

    Pass 2 — dependent hops (depends_on=<int>):
      resolve_template() substitutes {q0_answer} etc. with the concrete value
      from hop_results BEFORE extract_entities() runs.  This ensures the
      ChromaDB entity pre-filter activates on the correct institution name —
      the fix for Stage 4's parallel-execution limitation.

max_hops guard:
  sub-questions are truncated to cfg["agentic_rag"]["max_hops"] (default 3)
  before any retrieval runs, preventing latency compounding on deep chains.

Returns:
  Partial FinGraphState update: {"vector_chunks": [...], "graph_results": [...], "hop_results": {...}}
"""

from __future__ import annotations

import logging

from config import get_config
from rag._entities import extract_entities
from rag._reranker import rerank
from rag._resolver import has_template_vars, resolve_template
from rag._state import FinGraphState
from retrieval.graph_retriever import GraphResult
from retrieval.graph_retriever import retrieve as graph_retrieve
from retrieval.graph_retriever import retrieve_multihop
from retrieval.vector_retriever import retrieve as vector_retrieve

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_hop_answer(rows: list[dict]) -> str | None:
    """
    Extract the primary entity name from graph result rows for template substitution.

    Priority order for field names:
      i.name > name > institution_name > entity_name > first non-empty string value

    Returns None if no usable string value is found.
    """
    if not rows:
        return None
    first = rows[0]
    for key in ("i.name", "name", "institution_name", "entity_name"):
        val = first.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Fallback: first non-empty string value in the row
    for val in first.values():
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _run_vector_retrieval(
    q: str,
    cfg: dict,
    seen_chunks: set[str],
) -> list[str]:
    """
    Wide cosine retrieval → cross-encoder reranking → deduplicated formatted chunks.
    Extracts entity filter from *q* (post-template-resolution for dependent hops).
    """
    top_k       = cfg["agentic_rag"]["top_k_vector"]
    top_k_final = cfg["agentic_rag"]["top_k_vector_final"]

    entities = extract_entities(q)
    where: dict | None = None
    if entities:
        where = {"entity_name": {"$in": entities}}
        log.info("hop_executor vector | entity_filter=%s | %s", entities, q[:70])

    results = vector_retrieve(q, top_k=top_k, cfg=cfg, where=where)
    if not results:
        log.info("hop_executor vector | 0 chunks after filter | %s", q[:70])
        return []

    try:
        reranked = rerank(q, results, cfg, top_k=top_k_final)
    except Exception as exc:
        log.error("hop_executor rerank failed, using cosine top-%d: %s", top_k_final, exc)
        reranked = results[:top_k_final]

    chunks: list[str] = []
    for r in reranked:
        if r["text"] in seen_chunks:
            continue
        seen_chunks.add(r["text"])
        chunks.append(
            f"Entity: {r['entity_name']} | Date: {r['date']} | PR: {r['pr_number']}\n"
            f"{r['text']}"
        )

    log.info(
        "hop_executor vector | %d→%d chunks | entity_filter=%s | %s",
        len(results), len(chunks), where is not None, q[:70],
    )
    return chunks


# ── Main node ─────────────────────────────────────────────────────────────────

def hop_execute(state: FinGraphState) -> dict:
    """
    LangGraph node — serial multi-hop executor for Stage 5.

    Reads:  state["sub_questions"], state["routes"], state["hop_dependencies"],
            state["cypher_sufficient"], state["hop_results"] (empty on entry)
    Writes: vector_chunks, graph_results, hop_results
    """
    cfg = get_config()
    max_hops: int = cfg["agentic_rag"].get("max_hops", 3)

    # Truncate to max_hops guard
    sub_questions = state["sub_questions"][:max_hops]
    routes        = state["routes"][:max_hops]
    hop_deps      = state["hop_dependencies"][:max_hops]

    # ── cypher_sufficient=True path ───────────────────────────────────────────
    if state.get("cypher_sufficient", False):
        log.info("hop_executor | cypher_sufficient=True — attempting single-shot multi-hop Cypher")
        result = retrieve_multihop(sub_questions, cfg)
        if result["success"]:
            log.info("hop_executor | single-shot Cypher succeeded (%d rows)", len(result["rows"]))
            return {
                "graph_results": [result],
                "vector_chunks": [],
                "hop_results":   {},
            }
        # Cypher generation/execution failed — fall through to hybrid path
        log.warning(
            "hop_executor | single-shot Cypher failed — falling back to hybrid serial execution"
        )

    # ── cypher_sufficient=False path (hybrid serial) ──────────────────────────
    log.info("hop_executor | hybrid serial | %d hops (max_hops=%d)", len(sub_questions), max_hops)

    vector_chunks: list[str] = []
    graph_results: list[GraphResult] = []
    hop_results: dict = {}
    seen_chunks: set[str] = set()

    # ── Pass 1: independent hops (depends_on=None) ────────────────────────────
    for i, (q, route, dep) in enumerate(zip(sub_questions, routes, hop_deps)):
        if dep is not None:
            continue  # handled in pass 2

        log.info("hop_executor | hop %d (independent) | route=%s | %s", i, route, q[:70])

        # Graph retrieval
        if route in ("graph", "both"):
            gr = graph_retrieve(q, cfg=cfg)
            if gr["success"]:
                graph_results.append(gr)
                entity = _extract_hop_answer(gr["rows"])
                if entity:
                    hop_results[i] = entity
                    log.info("hop_executor | hop %d → entity='%s'", i, entity)
            else:
                log.info("hop_executor | hop %d graph FAIL | %s", i, q[:70])

        # Vector retrieval
        if route in ("vector", "both"):
            chunks = _run_vector_retrieval(q, cfg, seen_chunks)
            vector_chunks.extend(chunks)

    # ── Pass 2: dependent hops (depends_on=<int>) ─────────────────────────────
    for i, (q, route, dep) in enumerate(zip(sub_questions, routes, hop_deps)):
        if dep is None:
            continue  # already handled in pass 1

        # Template resolution — this is the core Stage 5 fix
        if dep in hop_results:
            resolved_q = resolve_template(q, hop_results)
            if resolved_q != q:
                log.info(
                    "hop_executor | hop %d | resolved '%s' → '%s'",
                    i, q[:60], resolved_q[:60],
                )
        else:
            log.warning(
                "hop_executor | hop %d depends on hop %d but hop %d has no result — "
                "running with unresolved template (entity filter will not activate)",
                i, dep, dep,
            )
            resolved_q = q

        log.info("hop_executor | hop %d (dep=%d) | route=%s | %s", i, dep, route, resolved_q[:70])

        # Graph retrieval on resolved question
        if route in ("graph", "both"):
            gr = graph_retrieve(resolved_q, cfg=cfg)
            if gr["success"]:
                graph_results.append(gr)
                entity = _extract_hop_answer(gr["rows"])
                if entity:
                    hop_results[i] = entity
                    log.info("hop_executor | hop %d → entity='%s'", i, entity)
            else:
                log.info("hop_executor | hop %d graph FAIL | %s", i, resolved_q[:70])

        # Vector retrieval on resolved question — entity filter now correct
        if route in ("vector", "both"):
            chunks = _run_vector_retrieval(resolved_q, cfg, seen_chunks)
            vector_chunks.extend(chunks)

    log.info(
        "hop_executor done | %d vector chunks | %d graph results | %d hop_results",
        len(vector_chunks), len(graph_results), len(hop_results),
    )
    return {
        "vector_chunks": vector_chunks,
        "graph_results": graph_results,
        "hop_results":   hop_results,
    }
