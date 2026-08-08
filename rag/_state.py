"""
rag/_state.py

Shared LangGraph state schema for Stage 3 Agentic GraphRAG.
Imported by agentic_rag.py and every module in rag/nodes/.

CLAUDE.md rule: LangGraph state must use FinGraphState(TypedDict).
Never pass raw dicts between LangGraph nodes.
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
    vector_chunks   Chunk texts accumulated by vector_node (RAGAS contexts).
    graph_results   Full GraphResult objects from graph_node -- retains the
                    Cypher query so the synthesiser can label each result group.
    context         Formatted merged context string consumed by synthesiser.
    answer          Final synthesised answer.
    iterations      Safety counter vs config agentic_rag.max_iterations.
    """

    question:      str
    sub_questions: list[str]
    routes:        list[str]         # one per sub-question: "vector"|"graph"|"both"
    vector_chunks: list[str]         # chunk texts from vector_retriever
    graph_results: list[GraphResult] # full GraphResult objects -- Cypher preserved
    context:       str               # merged formatted context for synthesiser
    answer:        str
    iterations:    int
