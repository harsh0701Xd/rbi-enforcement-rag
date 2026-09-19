"""
rag/nodes/hop_executor.py

LangGraph node: the engine of Stage 6 — owns ALL retrieval and multi-hop resolution.

Three execution paths (all branching inside this node; no conditional DAG edges):

  Path A: cypher_sufficient=True  (graph-only single-shot)
  ──────────────────────────────────────────────────────────
  The decomposer determined all sub-questions can be answered from Neo4j in one query.
  Calls retrieve_multihop() to generate and execute a single WITH-chained Cypher.
  hop_results[0] = rows[0] raw dict (for resolver/display).
  Falls back to hybrid serial (Path C) if Cypher generation/execution fails.

  Path B: fan_out=True  (iterate Q2 over all Q1 rows)
  ──────────────────────────────────────────────────────────
  Enables "for each X, what is Y?" query shapes.
  Runs Q1, extracts ALL result rows (capped at max_fan_out), then runs Q2 for each row
  by resolving {q0.field} placeholders with that row's data.
  Results from all Q2 iterations are collected into graph_results (sub_q_idx=1).

  Path C: default  (hybrid serial)
  ──────────────────────────────────────────────────────────
  Two-pass execution:
    Pass 1 — independent hops (depends_on=None): graph + vector retrieval.
    Pass 2 — dependent hops: resolve {q0.field} / {q0_answer} from hop_results,
              then run graph + vector on the resolved question.
  hop_results[i] = rows[0] raw dict — direct column lookup by {q0.field} resolver.
  Entity filter for vector uses the TEMPLATE question (pre-resolution) to prevent
  entity contamination (fix for Q5 bug: resolved entity name polluting ChromaDB filter).

Stage 6 key changes vs Stage 5:
  - _extract_hop_answer() removed — hop_results stores raw rows[0] dict, not a string
  - fan_out branch added (Path B)
  - hop_results[i] = gr["rows"][0] (raw dict) in all paths
  - _run_vector_retrieval() accepts filter_q param — entity filter from template question
  - resolve_template() handles both {q0.field} and {q0_answer} syntax via updated _resolver

Returns:
  Partial FinGraphState: {"vector_chunks": [...], "graph_results": [...], "hop_results": {...}}
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import get_config
from rag._resolver import has_template_vars, resolve_template
from rag._state import FinGraphState
from retrieval.graph_retriever import GraphResult
from retrieval.graph_retriever import retrieve as graph_retrieve
from retrieval.graph_retriever import retrieve_multihop
from retrieval.advanced_vector import retrieve_advanced

log = logging.getLogger(__name__)


# ── Vector retrieval helper ────────────────────────────────────────────────────

def _run_vector_retrieval(
    q: str,
    cfg: dict,
    seen_chunks: set[str],
    *,
    filter_q: str | None = None,
) -> list[str]:
    """
    Advanced vector retrieval (HyDE + metadata filter + cosine + date filter + rerank)
    → deduplicated formatted chunk strings ready for the synthesiser.

    Args:
        q:        Resolved query — used for HyDE hypothesis and cosine search.
        filter_q: If provided, extract metadata filters from this string instead of q.
                  Pass the pre-resolution template question in dependent hops to prevent
                  entity contamination when {q0.entity_name} substitution adds a spurious
                  institution to the ChromaDB filter.
        seen_chunks: Mutable set; already-seen chunk texts are skipped (deduplication).
    """
    reranked = retrieve_advanced(q, cfg, filter_q=filter_q)
    if not reranked:
        log.info("hop_executor vector | 0 chunks | %s", q[:70])
        return []

    chunks: list[str] = []
    for r in reranked:
        if r["text"] in seen_chunks:
            continue
        seen_chunks.add(r["text"])
        chunks.append(
            f"Entity: {r['entity_name']} | Date: {r['date']} | PR: {r['pr_number']}\n"
            f"{r['text']}"
        )

    log.info("hop_executor vector | %d chunks (deduped) | %s", len(chunks), q[:70])
    return chunks


# ── Main node ──────────────────────────────────────────────────────────────────

def hop_execute(state: FinGraphState) -> dict:
    """
    LangGraph node — the single retrieval engine for Stage 6.

    Reads:  state["sub_questions"], state["routes"], state["hop_dependencies"],
            state["cypher_sufficient"], state["fan_out"], state["hop_results"] (empty on entry)
    Writes: vector_chunks, graph_results, hop_results
    """
    cfg = get_config()
    max_hops: int    = cfg["agentic_rag"].get("max_hops", 3)
    max_fan_out: int = cfg["agentic_rag"].get("max_fan_out", 8)

    # Truncate to max_hops guard
    sub_questions = state["sub_questions"][:max_hops]
    routes        = state["routes"][:max_hops]
    hop_deps      = state["hop_dependencies"][:max_hops]

    vector_chunks: list[str]    = []
    graph_results: list[GraphResult] = []
    hop_results:   dict         = {}
    seen_chunks:   set[str]     = set()

    # ══════════════════════════════════════════════════════════════════════════
    # PATH A: cypher_sufficient=True — single-shot WITH-chained Cypher
    # ══════════════════════════════════════════════════════════════════════════
    if state.get("cypher_sufficient", False):
        log.info("hop_executor | PATH A: cypher_sufficient=True — single-shot multi-hop Cypher")
        result = retrieve_multihop(sub_questions, cfg)
        result["sub_q_idx"] = 0  # single-shot covers all hops; label as Q1
        if result["success"]:
            log.info("hop_executor | PATH A succeeded (%d rows)", len(result["rows"]))
            # Store first row as raw dict for resolver/display
            if result["rows"]:
                hop_results[0] = result["rows"][0]
            return {
                "graph_results": [result],
                "vector_chunks": [],
                "hop_results":   hop_results,
            }
        # Generation/execution failed — fall through to Path C (hybrid serial)
        log.warning("hop_executor | PATH A failed — falling back to Path C (hybrid serial)")

    # ══════════════════════════════════════════════════════════════════════════
    # PATH B: fan_out=True — iterate Q2 over all Q1 result rows
    # ══════════════════════════════════════════════════════════════════════════
    if state.get("fan_out", False) and len(sub_questions) >= 2:
        route1 = routes[1] if len(routes) > 1 else "graph"

        # ── PATH B graph-only: re-route to Path A (single WITH-chained Cypher) ──
        # A graph-only fan-out ("for each X how many Y") is just a WITH-chained
        # Cypher — running N separate graph queries is wasteful and identical to
        # what retrieve_multihop() already does in one round-trip.
        if route1 == "graph":
            log.info(
                "hop_executor | PATH B→A: fan_out=True but route=graph — "
                "delegating to retrieve_multihop (single WITH-chained Cypher)"
            )
            result = retrieve_multihop(sub_questions, cfg)
            result["sub_q_idx"] = 0
            if result["success"] and result["rows"]:
                hop_results[0] = result["rows"][0]
            return {
                "graph_results": [result],
                "vector_chunks": [],
                "hop_results":   hop_results,
            }

        # ── PATH B vector: iterate Q2 concurrently (vector needed per entity) ──
        log.info(
            "hop_executor | PATH B: fan_out=True | route=%s | Q1 graph then Q2 concurrent (cap=%d)",
            route1, max_fan_out,
        )

        # Run Q1 — always graph-only (Q1 produces the entity list)
        q0  = sub_questions[0]
        gr0 = graph_retrieve(q0, cfg=cfg)
        gr0["sub_q_idx"] = 0
        graph_results.append(gr0)

        if gr0["success"] and gr0["rows"]:
            hop_results[0] = gr0["rows"][0]
            q1_template = sub_questions[1]
            fan_rows    = gr0["rows"][:max_fan_out]

            log.info("hop_executor | PATH B: Q2 vector | %d rows | running concurrently", len(fan_rows))

            def _run_fan_out_row(row_idx: int, row: dict):
                """Execute one Q2 fan-out iteration (graph + vector)."""
                temp_hop    = {0: row}
                resolved_q1 = resolve_template(q1_template, temp_hop)
                row_graph: list = []
                row_chunks: list[str] = []

                if route1 in ("graph", "both"):
                    gr = graph_retrieve(resolved_q1, cfg=cfg)
                    gr["sub_q_idx"] = 1
                    row_graph.append(gr)

                if route1 in ("vector", "both"):
                    try:
                        chunks = _run_vector_retrieval(
                            resolved_q1, cfg, set(), filter_q=q1_template
                        )
                        row_chunks.extend(chunks)
                    except Exception as exc:
                        log.warning("hop_executor | PATH B row %d vector FAIL: %s", row_idx, exc)

                return row_graph, row_chunks

            # Concurrent execution — avoids N× sequential latency
            with ThreadPoolExecutor(max_workers=min(len(fan_rows), 4)) as pool:
                futures = {
                    pool.submit(_run_fan_out_row, idx, row): idx
                    for idx, row in enumerate(fan_rows)
                }
                for future in as_completed(futures):
                    row_idx = futures[future]
                    try:
                        row_graph, row_chunks = future.result()
                        graph_results.extend(row_graph)
                        for chunk in row_chunks:
                            if chunk not in seen_chunks:
                                seen_chunks.add(chunk)
                                vector_chunks.append(chunk)
                    except Exception as exc:
                        log.warning("hop_executor | PATH B row %d FAIL: %s", row_idx, exc)
        else:
            log.warning("hop_executor | PATH B: Q1 returned 0 rows or failed — no Q2 iterations")

        log.info(
            "hop_executor PATH B done | %d vector chunks | %d graph results",
            len(vector_chunks), len(graph_results),
        )
        return {
            "vector_chunks": vector_chunks,
            "graph_results": graph_results,
            "hop_results":   hop_results,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PATH C: default — hybrid serial (independent hops then dependent hops)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("hop_executor | PATH C: hybrid serial | %d hops (max_hops=%d)", len(sub_questions), max_hops)

    # ── Pass 1: independent hops (depends_on=None) ────────────────────────────
    for i, (q, route, dep) in enumerate(zip(sub_questions, routes, hop_deps)):
        if dep is not None:
            continue  # handled in Pass 2

        log.info("hop_executor | hop %d (independent) | route=%s | %s", i, route, q[:70])

        # Graph retrieval — keep ALL results (success AND failure) for UI stats
        if route in ("graph", "both"):
            gr = graph_retrieve(q, cfg=cfg)
            gr["sub_q_idx"] = i
            graph_results.append(gr)  # keep regardless of success
            if gr["success"] and gr["rows"]:
                hop_results[i] = gr["rows"][0]  # raw dict — used by {q0.field} resolver
                log.info("hop_executor | hop %d → rows[0]=%s", i, list(gr["rows"][0].keys()))
            else:
                log.info("hop_executor | hop %d graph FAIL | %s", i, q[:70])

        # Vector retrieval — no filter_q needed (q has no template vars in Pass 1)
        if route in ("vector", "both"):
            try:
                chunks = _run_vector_retrieval(q, cfg, seen_chunks)
                vector_chunks.extend(chunks)
            except Exception as exc:
                log.warning("hop_executor | hop %d vector FAIL (degrading to graph-only): %s", i, exc)

    # ── Pass 2: dependent hops (depends_on=<int>) ─────────────────────────────
    for i, (q, route, dep) in enumerate(zip(sub_questions, routes, hop_deps)):
        if dep is None:
            continue  # already handled in Pass 1

        # Resolve template — {q0.field} or {q0_answer} → concrete value from hop_results
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
                "running with unresolved template",
                i, dep, dep,
            )
            resolved_q = q

        log.info("hop_executor | hop %d (dep=%d) | route=%s | %s", i, dep, route, resolved_q[:70])

        # Graph retrieval — keep ALL results for UI stats
        if route in ("graph", "both"):
            gr = graph_retrieve(resolved_q, cfg=cfg)
            gr["sub_q_idx"] = i
            graph_results.append(gr)  # keep regardless of success
            if gr["success"] and gr["rows"]:
                hop_results[i] = gr["rows"][0]  # raw dict
                log.info("hop_executor | hop %d → rows[0]=%s", i, list(gr["rows"][0].keys()))
            else:
                log.info("hop_executor | hop %d graph FAIL | %s", i, resolved_q[:70])

        # Vector retrieval — filter_q = template question (pre-resolution)
        # This prevents entity contamination: e.g. resolved q "...in Axis Bank Limited?"
        # would add Axis Bank to HDFC Bank's vector filter if we used resolved_q.
        if route in ("vector", "both"):
            try:
                chunks = _run_vector_retrieval(
                    resolved_q, cfg, seen_chunks, filter_q=q  # q is the template, pre-resolution
                )
                vector_chunks.extend(chunks)
            except Exception as exc:
                log.warning("hop_executor | hop %d vector FAIL (degrading to graph-only): %s", i, exc)

    log.info(
        "hop_executor PATH C done | %d vector chunks | %d graph results | %d hop_results",
        len(vector_chunks), len(graph_results), len(hop_results),
    )
    return {
        "vector_chunks": vector_chunks,
        "graph_results": graph_results,
        "hop_results":   hop_results,
    }
