"""
retrieval/graph_retriever.py

Translates a natural-language question into Cypher via Claude Haiku,
executes it against Neo4j through execute_cypher_safe(), and returns
structured graph results.

Graph schema (Schema A — written by ingestion/graph_writer.py):
    (Institution)-[:RECEIVED]->(PenaltyAction)-[:FOR_VIOLATION]->(Violation)
                                              -[:CITES]->(Regulation)

Fallback chain (per CLAUDE.md §6 — non-negotiable):
  Haiku generates Cypher
      ↓  (CypherSyntaxError or empty result)
  Log error to cypher_errors.log
      ↓
  Return GraphResult(rows=[], success=False)
      ↓
  Caller (agentic_rag router) falls back to vector_retriever silently

Public API:
    GraphResult              — TypedDict for a graph query result
    retrieve(question, cfg)  — main retrieval function
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

import anthropic
from dotenv import load_dotenv
from neo4j.exceptions import CypherSyntaxError

from config import get_config
from ingestion._neo4j import build_driver, execute_cypher_safe

load_dotenv()

log = logging.getLogger(__name__)


# ── RBI enforcement graph schema ──────────────────────────────────────────────
# Must stay in sync with ingestion/graph_writer.py and _rbi_graph_queries.py.
from prompts.cypher import SCHEMA as _SCHEMA, PROMPT as _CYPHER_PROMPT, MULTIHOP_PROMPT as _MULTIHOP_PROMPT

# ── Aggregation intent detection ──────────────────────────────────────────────
# Queries that need COUNT/SUM/AVG must not be LIMIT-capped.
# Detected by keyword pattern; when matched the question is prefixed with
# "[AGGREGATION]" which triggers Rule 12 in the Cypher prompt.

_AGGREGATION_RE = re.compile(
    r'\bhow many\b|\btotal (?:penalty|penalties|fine|fines|amount)\b'
    r'|\bcount of\b|\bsum of\b|\baverage penalty\b|\boverall total\b'
    r'|\bhow much (?:in total|overall|combined)\b'
    r'|\bnumber of (?:banks|nbfcs|institutions|cases|actions)\b',
    re.IGNORECASE,
)


def _is_aggregation(question: str) -> bool:
    """Return True if the question requires aggregate Cypher (COUNT/SUM/AVG)."""
    return bool(_AGGREGATION_RE.search(question))



# ── Return type ───────────────────────────────────────────────────────────────

class GraphResult(TypedDict):
    cypher:  str         # generated Cypher (preserved for MLflow logging)
    rows:    list[dict]  # result rows; empty list on any failure
    success: bool        # False → caller must fall back to vector_retriever


# ── Module-level singletons ───────────────────────────────────────────────────

_driver:    object | None = None
_anthropic: anthropic.Anthropic | None = None


def _get_driver(cfg: dict):
    global _driver
    if _driver is None:
        _driver = build_driver(cfg)
    return _driver


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = anthropic.Anthropic()
    return _anthropic


# ── Cypher generation ─────────────────────────────────────────────────────────

def _generate_cypher(question: str, cfg: dict) -> str | None:
    """
    Ask Claude (cypher_model if configured, else system_model) to produce a Cypher query.
    Temperature = 0 (cfg["llm"]["cypher_temperature"]) — deterministic output.
    Returns None on any generation failure so retrieve() falls back cleanly.

    Stage 6: uses cfg["models"]["cypher_model"] (Sonnet) for better Cypher reasoning.
    Falls back to system_model if cypher_model is not configured.
    """
    # Prefix aggregation queries so Rule 12 fires in the Cypher prompt
    q_for_prompt = ("[AGGREGATION] " + question) if _is_aggregation(question) else question

    prompt = _CYPHER_PROMPT.format(
        schema   = _SCHEMA,
        limit    = cfg["agentic_rag"]["cypher_limit"],
        question = q_for_prompt,
    )
    # Stage 6: separate model for Cypher generation (better at structured query reasoning)
    cypher_model = cfg["models"].get("cypher_model", cfg["models"]["system_model"])
    try:
        msg = _get_anthropic().messages.create(
            model       = cypher_model,
            max_tokens  = cfg["llm"]["max_tokens_cypher"],
            temperature = cfg["llm"]["cypher_temperature"],
            messages    = [{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences if the model wraps output despite instructions
        raw = re.sub(r"^```(?:cypher)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()
        return raw or None
    except Exception as exc:
        log.error("Cypher generation failed: %s", exc)
        return None


# ── Main retrieval function ───────────────────────────────────────────────────

def retrieve(question: str, cfg: dict | None = None) -> GraphResult:
    """
    Full NL → Cypher → Neo4j pipeline for RBI enforcement queries.

    Never raises — all failures return GraphResult(rows=[], success=False)
    so the agentic_rag router can fall back to vector_retriever without
    surfacing errors to the user.

    Args:
        question: Natural-language question or decomposed sub-question.
        cfg:      Config dict from get_config(). Loaded automatically if None.

    Returns:
        GraphResult with rows populated on success, empty on any failure.
    """
    if cfg is None:
        cfg = get_config()

    cypher = _generate_cypher(question, cfg)
    if not cypher:
        return GraphResult(cypher="", rows=[], success=False)

    log.info("Generated Cypher | %s", cypher.replace("\n", " "))

    try:
        rows = execute_cypher_safe(cypher, _get_driver(cfg))
    except CypherSyntaxError as exc:
        log.warning("CypherSyntaxError → falling back to vector | %s", exc)
        return GraphResult(cypher=cypher, rows=[], success=False)
    except Exception as exc:
        log.error("Neo4j execution error: %s", exc)
        return GraphResult(cypher=cypher, rows=[], success=False)

    log.info("Graph query returned %d rows", len(rows))
    return GraphResult(cypher=cypher, rows=rows, success=True)


# ── Multi-hop retrieval (cypher_sufficient=True path) ─────────────────────────

def _generate_multihop_cypher(sub_questions: list[str], cfg: dict) -> str | None:
    """
    Ask Claude (cypher_model) to generate a single WITH-chained Cypher query
    that covers all sub-questions in one DB round-trip.

    Stage 6: uses cypher_model (Sonnet) — better handling of WITH alias rules
    and complex multi-hop chaining.  Falls back to system_model if not configured.

    Called only when cypher_sufficient=True — i.e. the decomposer determined
    that all hops are answerable from the graph without vector retrieval.
    """
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))
    prompt = _MULTIHOP_PROMPT.format(
        schema        = _SCHEMA,
        limit         = cfg["agentic_rag"]["cypher_limit"],
        sub_questions = numbered,
    )
    cypher_model = cfg["models"].get("cypher_model", cfg["models"]["system_model"])
    try:
        msg = _get_anthropic().messages.create(
            model       = cypher_model,
            max_tokens  = cfg["llm"]["max_tokens_cypher"],
            temperature = cfg["llm"]["cypher_temperature"],
            messages    = [{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```(?:cypher)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()
        return raw or None
    except Exception as exc:
        log.error("Multi-hop Cypher generation failed: %s", exc)
        return None


def retrieve_multihop(sub_questions: list[str], cfg: dict | None = None) -> GraphResult:
    """
    Generate and execute a single WITH-chained Cypher query covering all
    sub_questions in one Neo4j round-trip.

    Called by hop_executor when cypher_sufficient=True.  Never raises —
    returns GraphResult(rows=[], success=False) on any failure so the caller
    can fall back to hybrid serial execution.

    Args:
        sub_questions: Ordered list of atomic sub-questions (already template-free;
                       cypher_sufficient=True means no vector hops are needed).
        cfg:           Config dict from get_config(). Loaded automatically if None.

    Returns:
        GraphResult with rows populated on success, empty on failure.
    """
    if cfg is None:
        cfg = get_config()

    cypher = _generate_multihop_cypher(sub_questions, cfg)
    if not cypher:
        return GraphResult(cypher="", rows=[], success=False)

    log.info("Multi-hop Cypher generated | %s", cypher.replace("\n", " ")[:200])

    try:
        rows = execute_cypher_safe(cypher, _get_driver(cfg))
    except CypherSyntaxError as exc:
        log.warning("Multi-hop CypherSyntaxError → falling back | %s", exc)
        return GraphResult(cypher=cypher, rows=[], success=False)
    except Exception as exc:
        log.error("Multi-hop Neo4j execution error: %s", exc)
        return GraphResult(cypher=cypher, rows=[], success=False)

    log.info("Multi-hop Cypher returned %d rows", len(rows))
    return GraphResult(cypher=cypher, rows=rows, success=True)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    _cfg = get_config()
    tests = [
        "What penalty did RBI impose on Axis Bank?",
        "Which NBFCs were penalised by RBI?",
        "Which banks violated KYC directions in 2023?",
        "What regulations did cooperative banks violate?",
    ]
    for q in tests:
        r = retrieve(q, cfg=_cfg)
        print(f"\nQ: {q}")
        print(f"   Cypher : {r['cypher'][:120].replace(chr(10), ' ')}")
        print(f"   Success: {r['success']}  |  Rows: {len(r['rows'])}")
        for row in r["rows"][:3]:
            print(f"   → {row}")
