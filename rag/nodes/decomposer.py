"""
rag/nodes/decomposer.py

LangGraph node: decompose the user question into at most max_sub_questions
atomic sub-questions using Claude Haiku at temperature=0, and classify each
sub-question's retrieval route (vector | graph | both).

Routing is computed here using the deterministic keyword + named-entity
classifier from rag/nodes/router.py (_classify).  The router node has been
removed from the LangGraph DAG; this node now populates both sub_questions
and routes in a single pass, saving one LLM round-trip.

Stage 6 changes:
  - execution_mode removed from prompt output (always "serial" in Stage 6)
  - fan_out added: True when Q2 must iterate over ALL Q1 result rows
  - placeholder syntax updated: {q0.field_name} instead of {q0_answer}
  - _parse_json now returns (subs, exec_mode, cypher_ok, deps, fan_out)

Fallback behaviour (robust against any LLM/parse failure):
  - If JSON parsing fails → single-item [original_question], serial, no fan_out
  - Routes default to ["graph"] on fallback
  - All state fields get safe defaults on fallback
"""

from __future__ import annotations

import json
import logging
import re

import anthropic
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_config
from rag._state import FinGraphState

load_dotenv()

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


from prompts.decomposer import PROMPT as _PROMPT
from rag.nodes.router import _classify


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    reraise=True,
)
def _call(question: str, cfg: dict) -> str:
    max_q: int = cfg["agentic_rag"]["max_sub_questions"]
    msg = _get_client().messages.create(
        model       = cfg["models"]["system_model"],
        max_tokens  = 512,
        temperature = cfg["llm"]["chat_temperature"],
        messages    = [{"role": "user", "content": _PROMPT.format(
            max_q=max_q, question=question,
        )}],
    )
    return msg.content[0].text.strip()


_VALID_ROUTES = {"graph", "vector", "both"}


def _parse_json(raw: str, question: str, max_q: int) -> tuple[list[str], str, bool, list, bool, list[str] | None]:
    """
    Parse LLM JSON output into
    (sub_questions, execution_mode, cypher_sufficient, hop_deps, fan_out, llm_routes).

    llm_routes is a list[str] when the LLM produced valid routes, or None when
    routes are absent/invalid (caller falls back to _classify).

    Stage 6: execution_mode is always "serial" (all queries through hop_executor).
    fan_out is a new boolean field — true when Q2 should iterate over all Q1 rows.

    Strips markdown fences if present.  Returns safe fallback values on any parse error.
    """
    # Strip markdown code fences the model may emit despite the prompt
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("Decomposer JSON parse error (%s) — falling back to single-hop serial", exc)
        return [question], "serial", False, [None], False, None

    # Validate structure
    if not isinstance(data, dict) or "sub_questions" not in data:
        log.warning("Decomposer response missing 'sub_questions' key — falling back")
        return [question], "serial", False, [None], False, None

    raw_subs = data.get("sub_questions", [])
    if not isinstance(raw_subs, list) or not raw_subs:
        log.warning("Decomposer 'sub_questions' is empty or not a list — falling back")
        return [question], "serial", False, [None], False, None

    # Extract sub-question texts and dependency list (truncate to max_q)
    subs: list[str] = []
    deps: list      = []
    for item in raw_subs[:max_q]:
        q_text = item.get("q", "").strip()
        dep    = item.get("depends_on")  # None or int
        if not q_text:
            continue
        subs.append(q_text)
        deps.append(None if dep is None else int(dep))

    if not subs:
        return [question], "serial", False, [None], False, None

    cypher_ok: bool = bool(data.get("cypher_sufficient", False))
    fan_out:   bool = bool(data.get("fan_out", False))

    # Sanity: fan_out and cypher_sufficient are mutually exclusive
    if fan_out and cypher_ok:
        log.warning(
            "Decomposer returned fan_out=True with cypher_sufficient=True — "
            "fan-out requires multiple queries; forcing cypher_sufficient=False"
        )
        cypher_ok = False

    # In Stage 6 all paths go through hop_executor — execution_mode is always "serial"
    exec_mode = "serial"

    # Parse routes[] — validate length and values; None signals fallback to _classify
    llm_routes: list[str] | None = None
    raw_routes = data.get("routes")
    if (
        isinstance(raw_routes, list)
        and len(raw_routes) == len(subs)
        and all(r in _VALID_ROUTES for r in raw_routes)
    ):
        llm_routes = raw_routes
    elif raw_routes is not None:
        log.warning(
            "Decomposer routes invalid (got %r for %d subs) — falling back to _classify",
            raw_routes, len(subs),
        )

    log.debug(
        "_parse_json: subs=%s deps=%s cypher_ok=%s fan_out=%s routes=%s",
        subs, deps, cypher_ok, fan_out, llm_routes,
    )
    return subs, exec_mode, cypher_ok, deps, fan_out, llm_routes


def decompose(state: FinGraphState) -> dict:
    """
    LangGraph node — decompose state["question"] into sub-questions.

    Returns a partial state update with:
      sub_questions, routes, iterations, execution_mode,
      cypher_sufficient, hop_dependencies, fan_out
    """
    cfg      = get_config()
    question = state["question"]
    max_q: int = cfg["agentic_rag"]["max_sub_questions"]

    try:
        raw = _call(question, cfg)
        subs, exec_mode, cypher_ok, deps, fan_out, llm_routes = _parse_json(raw, question, max_q)
    except Exception as exc:
        log.error("Decomposer failed — using original question as single serial hop: %s", exc)
        subs       = [question]
        exec_mode  = "serial"
        cypher_ok  = False
        deps       = [None]
        fan_out    = False
        llm_routes = None

    # Use LLM-provided routes when valid; fall back to keyword classifier otherwise.
    if llm_routes is not None:
        routes = llm_routes
        route_source = "LLM"
    else:
        routes = [_classify(q) for q in subs]
        route_source = "classifier"

    log.info(
        "Decomposed into %d sub-question(s) | cypher_sufficient=%s | fan_out=%s | deps=%s | routes=%s [%s]",
        len(subs), cypher_ok, fan_out, deps, routes, route_source,
    )
    for i, (q, d, r) in enumerate(zip(subs, deps, routes)):
        log.debug("  hop %d (depends_on=%s, route=%s): %s", i, d, r, q)

    return {
        "sub_questions":     subs,
        "routes":            routes,
        "iterations":        state.get("iterations", 0) + 1,
        "execution_mode":    exec_mode,
        "cypher_sufficient": cypher_ok,
        "hop_dependencies":  deps,
        "fan_out":           fan_out,
    }
