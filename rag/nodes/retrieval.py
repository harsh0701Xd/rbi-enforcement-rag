"""
rag/nodes/retrieval.py

Two LangGraph nodes -- vector_node and graph_node -- that run retrieval for
every sub-question matching their assigned route.

vector_node: sub-questions routed "vector" or "both"  [Stage 4 upgrades]
  1. Entity_name pre-filter  -- extract_entities() narrows ChromaDB to the
     named institution before cosine search.
  2. Wider retrieval         -- cfg["agentic_rag"]["top_k_vector"] (10) chunks
     retrieved per sub-question (was cfg["naive_rag"]["top_k"] = 5).
  3. Cross-encoder reranking -- rerank() scores (sub-question, chunk) pairs
     and keeps cfg["agentic_rag"]["top_k_vector_final"] (5) per sub-question.
  -> appends formatted chunk strings to state (deduped across sub-questions)

graph_node:  sub-questions routed "graph"  or "both"
  -> retrieval.graph_retriever.retrieve()  -> appends full GraphResult objects
     (GraphResult.cypher is preserved so the synthesiser can label each result
     group with its semantic context, e.g. "investors who backed PhonePe").

Both nodes accumulate results across all sub-questions before returning.
graph_node never raises -- GraphResult.success=False is silently skipped
(fallback chain: graph fail -> synthesiser uses vector_chunks only).
"""

from __future__ import annotations

import logging

from config import get_config
from rag._entities import extract_entities
from rag._reranker import rerank
from rag._state import FinGraphState
from retrieval.graph_retriever import GraphResult, retrieve as graph_retrieve
from retrieval.vector_retriever import retrieve as vector_retrieve

log = logging.getLogger(__name__)


def vector_node(state: FinGraphState) -> dict:
    """
    LangGraph node -- vector retrieval for sub-questions routed "vector" or "both".

    Stage 4 upgrades vs Stage 3:
      - Entity_name pre-filter narrows ChromaDB to the named institution.
      - top_k=10 wide retrieval (was 5, borrowed from naive_rag).
      - Cross-encoder reranking to top_k_final=5 per sub-question.

    Returns: {"vector_chunks": [formatted_chunk_string, ...]}
    Chunk strings include entity/date/PR metadata prepended so the synthesiser
    and RAGAS context_entity_recall can see canonical entity names directly.
    """
    cfg         = get_config()
    top_k       = cfg["agentic_rag"]["top_k_vector"]
    top_k_final = cfg["agentic_rag"]["top_k_vector_final"]

    chunks: list[str] = []
    seen:   set[str]  = set()   # dedup key: raw text (avoids metadata drift)

    for q, route in zip(state["sub_questions"], state["routes"]):
        if route not in ("vector", "both"):
            continue

        # Entity_name pre-filter -- narrows ChromaDB before cosine search.
        # Falls back to full-corpus scan when no registered entity is found
        # (unknown long-tail institutions use the fallback regex for routing
        # but have no canonical form safe to filter on).
        where: dict | None = None
        entities = extract_entities(q)
        if entities:
            where = {"entity_name": {"$in": entities}}
            log.info("vector_node | entity filter: %s | %s", entities, q[:60])

        # Wide cosine retrieval
        results = vector_retrieve(q, top_k=top_k, cfg=cfg, where=where)
        if not results:
            log.info("vector_node | %-6s | 0 chunks after filter | %s", route, q[:60])
            continue

        # Cross-encoder reranking per sub-question
        try:
            reranked = rerank(q, results, cfg, top_k=top_k_final)
        except Exception as exc:
            log.error("vector_node rerank failed, using cosine top-%d: %s", top_k_final, exc)
            reranked = results[:top_k_final]

        for r in reranked:
            if r["text"] in seen:
                continue
            seen.add(r["text"])
            # Prepend metadata so synthesiser and RAGAS see canonical entity names
            formatted = (
                f"Entity: {r['entity_name']} | Date: {r['date']} | PR: {r['pr_number']}\n"
                f"{r['text']}"
            )
            chunks.append(formatted)

        log.info(
            "vector_node | %-6s | %d→%d chunks | entity_filter=%s | %s",
            route, len(results), len(reranked), where is not None, q[:60],
        )

    log.info("vector_node total: %d unique chunks", len(chunks))
    return {"vector_chunks": chunks}


def graph_node(state: FinGraphState) -> dict:
    """
    LangGraph node -- graph retrieval for sub-questions routed "graph" or "both".
    Stores full GraphResult objects (including Cypher) so the synthesiser can
    label each result group with semantic context.

    ALL results (success AND failure) are now appended so the API can surface
    n_graph_attempted vs n_graph_succeeded to the UI.  Failed results have
    rows=[] and their rows are not passed to the synthesiser
    (_graph_rows_to_context_strings filters for success=True).

    Each result is tagged with sub_q_idx (the index within sub_questions) so
    the UI can label per-hop Cyphers ("Q1 Cypher", "Q2 Cypher", …).

    Returns: {"graph_results": [GraphResult, ...]}
    """
    cfg = get_config()
    results: list[GraphResult] = []

    for i, (q, route) in enumerate(zip(state["sub_questions"], state["routes"])):
        if route not in ("graph", "both"):
            continue
        result = graph_retrieve(q, cfg=cfg)
        result["sub_q_idx"] = i          # tag for UI labelling (not in TypedDict but safe)
        results.append(result)           # keep ALL — success AND failure
        if result["success"]:
            log.info("graph_node | %-6s | %d rows | %s", route, len(result["rows"]), q)
        else:
            log.info("graph_node | %-6s | FAIL (fallback to vector) | %s", route, q)

    n_ok = sum(1 for r in results if r.get("success"))
    log.info("graph_node total: %d attempted, %d succeeded", len(results), n_ok)
    return {"graph_results": results}
