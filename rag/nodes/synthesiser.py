"""
rag/nodes/synthesiser.py

LangGraph node: aggregate vector chunks + graph results into a unified context
string, then call Claude Haiku to synthesise the final answer.

Context formatting:
  Vector chunks  -> [V1] Entity: ... | Date: ... | PR: ...\n chunk_text
  Graph results  -> [Graph: <Cypher summary>]
                    [1] key: val, key: val   (one line per result row)

Including the Cypher query as a label is critical -- it tells Haiku what the
rows represent (e.g. "penalties received by Axis Bank") so it can synthesise
correctly rather than treating bare rows as opaque data.

Citation rule (same as Stages 1 and 2):
  Whenever a fact is drawn from a graph row or vector chunk, cite inline as
  (Entity Name, YYYY-MM-DD, PR: <pr_number>).
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from config import get_config
from rag._state import FinGraphState
from retrieval.graph_retriever import GraphResult

load_dotenv()

log = logging.getLogger(__name__)

from prompts.synthesis import AGENTIC_SYSTEM as _SYSTEM
from rag._synthesis import synthesise as _call_haiku


def _format_row(row: dict) -> str:
    """Format a Neo4j result row as 'key: value' pairs, stripping alias prefixes."""
    parts = [
        f"{k.split('.')[-1]}: {v}"
        for k, v in row.items()
        if v is not None
    ]
    return ", ".join(parts)


def _build_context(vector_chunks: list[str], graph_results: list[GraphResult]) -> str:
    """
    Merge vector chunks and graph results into a single numbered context block.

    Each GraphResult is labelled with its Cypher query (first 120 chars) so
    Haiku understands what relationship the rows represent — without this,
    rows like 'name: Axis Bank, penalty_lakh: 5.0' have no connection to the question.
    """
    passages: list[str] = []

    for i, text in enumerate(vector_chunks, 1):
        passages.append(f"[V{i}] {text}")

    for gr in graph_results:
        if not gr["rows"]:
            continue
        cypher_label = gr["cypher"].replace("\n", " ").strip()[:400]
        block = [f"[Graph query: {cypher_label}]"]
        for j, row in enumerate(gr["rows"], 1):
            formatted = _format_row(row)
            if formatted:
                block.append(f"  [{j}] {formatted}")
        passages.append("\n".join(block))

    return "\n\n".join(passages) if passages else "(no context retrieved)"



def synthesise(state: FinGraphState) -> dict:
    """
    LangGraph node -- build unified context then synthesise final answer.
    Returns: {"context": str, "answer": str}
    """
    cfg     = get_config()
    context = _build_context(state["vector_chunks"], state["graph_results"])

    total_rows = sum(len(gr["rows"]) for gr in state["graph_results"])
    log.info(
        "Synthesiser | vector_chunks=%d | graph_results=%d (%d rows)",
        len(state["vector_chunks"]), len(state["graph_results"]), total_rows,
    )

    try:
        answer = _call_haiku(context, state["question"], cfg, _SYSTEM)
    except Exception as exc:
        log.error("Synthesis failed: %s", exc)
        answer = "An error occurred while generating the answer. Please try again."

    log.info("Answer generated | chars=%d", len(answer))
    return {"context": context, "answer": answer}
