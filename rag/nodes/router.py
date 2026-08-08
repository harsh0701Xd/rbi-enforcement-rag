"""
rag/nodes/router.py

LangGraph node: classify each sub-question as "vector" | "graph" | "both".

Routing logic (in priority order):
  1. Named entity detected     -> "both"   (graph for structured lookup +
                                             vector for narrative context)
  2. Graph AND vector keywords -> "both"   (compound conceptual + factual query)
  3. Vector keywords only      -> "vector" (definitional, explanatory)
  4. Default                   -> "graph"  (most RBI queries are entity/relational)

Named entity detection uses a two-layer approach:
  • Gazetteer: ~60 known RBI-regulated institution names (case-insensitive)
  • Regex fallback: patterns like "X Bank", "X Finance", "X Co-operative Bank"
    that catch long-tail institution names not in the gazetteer

Why "both" for named entities?
  The graph gives structured penalty rows (penalty_lakh, date, violation text).
  The vector gives narrative press-release passages (richer regulatory language).
  Together they produce better-cited, more complete answers than either alone.
"""

from __future__ import annotations

import logging

from rag._entities import has_named_entity as _has_named_entity
from rag._state import FinGraphState

log = logging.getLogger(__name__)


# ── Keyword sets ──────────────────────────────────────────────────────────────

_GRAPH_KW = frozenset({
    # Enforcement actions
    "penalty", "penalised", "penalized", "penalise", "penalize",
    "fine", "fined", "imposed", "imposition",
    "action", "enforcement", "sanctions",
    # Registration / licence
    "cancelled", "cancel", "cancellation", "cor",
    "certificate of registration",
    "licence", "license", "revoked", "revoke", "suspended", "suspension",
    # Query patterns that need structured lookup
    "which bank", "which banks", "which nbfc", "which nbfcs",
    "which institution", "which entity",
    "how many", "how much", "total penalty", "total fine",
    "list of", "list all",
    # Domain-specific relational terms
    "violated", "violation", "non-compliance", "non compliance",
    "directions", "rbi imposed", "reserve bank imposed",
    "received penalty", "received fine",
    # Temporal + factual
    "in 2021", "in 2022", "in 2023", "in 2024", "in 2025",
    "during 2021", "during 2022", "during 2023", "during 2024",
    "from 2021", "from 2022", "from 2023",
})

_VECTOR_KW = frozenset({
    "what is", "what are", "explain", "describe", "define", "definition",
    "how does", "why does", "what does", "overview", "introduction",
    "history", "background", "process", "procedure", "mechanism",
    "policy", "framework", "regulation says", "directive says",
    "typical", "common", "generally", "usually", "in general",
    "difference between", "compare",
})


# ── Classification ────────────────────────────────────────────────────────────

def _classify(question: str) -> str:
    """
    Return "vector" | "graph" | "both" for a single question.

    Priority:
      1. Named entity present → "both" (graph lookup + vector context)
      2. Graph + vector keywords → "both"
      3. Vector keywords only  → "vector"
      4. Default               → "graph"
    """
    if _has_named_entity(question):
        log.debug("Named entity detected → both | %s", question[:80])
        return "both"

    ql = question.lower()
    has_graph  = any(kw in ql for kw in _GRAPH_KW)
    has_vector = any(kw in ql for kw in _VECTOR_KW)

    if has_graph and has_vector:
        return "both"
    if has_vector:
        return "vector"
    return "graph"   # default for RBI domain


def route(state: FinGraphState) -> dict:
    """
    LangGraph node — classify each sub-question into a retrieval route.
    Returns: {"routes": ["vector"|"graph"|"both", ...]}
    """
    routes = [_classify(q) for q in state["sub_questions"]]
    for q, r in zip(state["sub_questions"], routes):
        log.info("Route %-6s | %s", r, q)
    return {"routes": routes}
