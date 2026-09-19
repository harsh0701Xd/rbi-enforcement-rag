"""
rag/_state.py

Shared LangGraph state schema for Stage 6 Agentic GraphRAG.
Imported by agentic_rag.py and every module in rag/nodes/.

CLAUDE.md rule: LangGraph state must use FinGraphState(TypedDict).
Never pass raw dicts between LangGraph nodes.

Stage 6 changes:
  - execution_mode kept for UI backward-compat (always "serial" in Stage 6)
  - hop_results changed: int -> dict (raw rows[0]) instead of int -> str
  - fan_out added: True when hop_executor iterates Q2 over all Q1 result rows
"""

from __future__ import annotations

from typing import TypedDict

from retrieval.graph_retriever import GraphResult


class FinGraphState(TypedDict):
    """
    Typed state passed between LangGraph nodes.
    Each node returns a PARTIAL dict of updates; LangGraph merges automatically.

    question        Original user question -- never modified after graph entry.
    sub_questions   Atomic sub-questions produced by the decomposer node.
    routes          Routing decision per sub-question: "vector"|"graph"|"both".
    vector_chunks   Chunk texts accumulated by hop_executor (RAGAS contexts).
    graph_results   Full GraphResult objects from hop_executor -- retains the
                    Cypher query so the synthesiser can label each result group.
    context         Formatted merged context string consumed by synthesiser.
    answer          Final synthesised answer.
    iterations      Safety counter vs config agentic_rag.max_iterations.

    Stage 5/6 additions — set by decomposer, read by hop_executor:
      execution_mode    Always "serial" in Stage 6 (kept for UI backward-compat).
      cypher_sufficient True → single WITH-chained Cypher covers all hops.
      hop_dependencies  per-hop dependency: None | int (id of hop this depends on).
      hop_results       hop_id (int) -> raw rows[0] dict from that hop's Cypher result.
                        Stage 6: dict value, not string — use {q0.field} resolver.

    Stage 6 additions:
      fan_out           True when hop_executor should iterate Q2 over ALL Q1 result
                        rows (capped at max_fan_out). Enables "for each X, what is Y?"
                        query shapes.
    """

    question:      str
    sub_questions: list[str]
    routes:        list[str]          # one per sub-question: "vector"|"graph"|"both"
    vector_chunks: list[str]          # chunk texts from vector_retriever
    graph_results: list[GraphResult]  # full GraphResult objects -- Cypher preserved
    context:       str                # merged formatted context for synthesiser
    answer:        str
    iterations:    int
    # Stage 5/6 -- set by decomposer, read by agentic_rag + hop_executor
    execution_mode:    str   # always "serial" in Stage 6 (kept for UI compat)
    cypher_sufficient: bool  # True → single-shot WITH-chained Cypher covers all hops
    hop_dependencies:  list  # per-hop dependency: None | int
    hop_results:       dict  # hop_id (int) -> raw rows[0] dict (Stage 6)
    # Stage 6 only
    fan_out: bool            # True → iterate Q2 over all Q1 result rows
