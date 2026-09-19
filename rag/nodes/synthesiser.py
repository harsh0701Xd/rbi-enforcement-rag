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


# ── INR formatting ─────────────────────────────────────────────────────────────

def format_inr(lakh: float) -> str:
    """Convert a lakh float to an Indian-formatted rupee string.

    Examples:
      0.05  → ₹5,000
      25.0  → ₹25,00,000
      200.0 → ₹2,00,00,000
      578.55 → ₹5,78,55,000
    """
    rupees = round(lakh * 100_000)
    s = str(rupees)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    groups: list[str] = []
    while rest:
        groups.append(rest[-2:])
        rest = rest[:-2]
    groups.reverse()
    return f"₹{','.join(groups)},{last3}"


def _format_row(row: dict) -> str:
    """Format a Neo4j result row as 'key: value' pairs, stripping alias prefixes.

    Any key ending in '_lakh' is treated as a lakh-denominated amount and
    converted to an Indian-formatted rupee string. The '_lakh' suffix is
    stripped from the display key so the LLM sees e.g. 'total_penalty: ₹49,73,48,000'
    rather than 'total_penalty_lakh: 4973.48'.
    """
    parts = []
    for k, v in row.items():
        if v is None:
            continue
        short_k = k.split(".")[-1]
        if short_k.endswith("_lakh") and isinstance(v, (int, float)):
            display_k = short_k[:-5]  # strip "_lakh" suffix
            parts.append(f"{display_k}: {format_inr(float(v))}")
        else:
            parts.append(f"{short_k}: {v}")
    return ", ".join(parts)


_MAX_GRAPH_ROWS = 150  # cap per graph result to stay well under the 200K token limit


def _build_context(vector_chunks: list[str], graph_results: list[GraphResult]) -> str:
    """
    Merge vector chunks and graph results into a single numbered context block.

    Each GraphResult is labelled with its Cypher query (first 120 chars) so
    Haiku understands what relationship the rows represent — without this,
    rows like 'name: Axis Bank, penalty_lakh: 5.0' have no connection to the question.

    Graph rows are capped at _MAX_GRAPH_ROWS per result to prevent context overflow
    on large aggregation queries (e.g. "most frequent citation" returns 2000+ rows).
    """
    passages: list[str] = []

    for i, text in enumerate(vector_chunks, 1):
        passages.append(f"[V{i}] {text}")

    for gr in graph_results:
        cypher_label = gr["cypher"].replace("\n", " ").strip()[:400]
        rows = gr["rows"]
        if not rows:
            # Zero-row result — include the query label so synthesis knows the query ran
            # and can correctly interpret the empty result (e.g. NOT EXISTS → all matched)
            block = [
                f"[Graph query: {cypher_label}]",
                "  (Query executed successfully — 0 rows returned. If this query used"
                " NOT EXISTS, IS NULL, or absence logic, interpret this as: no entities"
                " matched the absence condition, meaning ALL entities of the filtered"
                " type have the relationship or property in question.)",
            ]
            passages.append("\n".join(block))
            continue
        truncated = len(rows) > _MAX_GRAPH_ROWS
        display_rows = rows[:_MAX_GRAPH_ROWS]
        block = [f"[Graph query: {cypher_label}]"]
        for j, row in enumerate(display_rows, 1):
            formatted = _format_row(row)
            if formatted:
                block.append(f"  [{j}] {formatted}")
        if truncated:
            block.append(f"  ... ({len(rows) - _MAX_GRAPH_ROWS} more rows not shown)")
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
