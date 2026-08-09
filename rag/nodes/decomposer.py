"""
rag/nodes/decomposer.py

LangGraph node: decompose the user question into at most max_sub_questions
atomic sub-questions using Claude Haiku at temperature=0.

Stage 5 change: output is now JSON that includes execution_mode, cypher_sufficient,
and per-hop dependency info (depends_on). The node populates the new FinGraphState
fields added in Stage 5.

Fallback behaviour (robust against any LLM/parse failure):
  - If JSON parsing fails → treat as parallel, single-item [original_question]
  - All 4 new state fields get safe defaults on fallback
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
        max_tokens  = 512,                             # increased from 256 for JSON output
        temperature = cfg["llm"]["chat_temperature"],
        messages    = [{"role": "user", "content": _PROMPT.format(
            max_q=max_q, question=question,
        )}],
    )
    return msg.content[0].text.strip()


def _parse_json(raw: str, question: str, max_q: int) -> tuple[list[str], str, bool, list]:
    """
    Parse the LLM JSON output into (sub_questions, execution_mode, cypher_sufficient, hop_deps).

    Strips markdown fences if present.  Returns safe fallback values on any parse error.
    """
    # Strip markdown code fences the model may emit despite the prompt
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("Decomposer JSON parse error (%s) — falling back to parallel single-hop", exc)
        return [question], "parallel", False, [None]

    # Validate structure
    if not isinstance(data, dict) or "sub_questions" not in data:
        log.warning("Decomposer response missing 'sub_questions' key — falling back")
        return [question], "parallel", False, [None]

    raw_subs = data.get("sub_questions", [])
    if not isinstance(raw_subs, list) or not raw_subs:
        log.warning("Decomposer 'sub_questions' is empty or not a list — falling back")
        return [question], "parallel", False, [None]

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
        return [question], "parallel", False, [None]

    exec_mode: str = data.get("execution_mode", "parallel")
    if exec_mode not in ("parallel", "serial"):
        exec_mode = "parallel"

    cypher_ok: bool = bool(data.get("cypher_sufficient", False))

    # Sanity check: if any dep is non-None, mode must be serial
    if any(d is not None for d in deps) and exec_mode == "parallel":
        log.warning("Decomposer returned dependencies with parallel mode — forcing serial")
        exec_mode = "serial"

    return subs, exec_mode, cypher_ok, deps


def decompose(state: FinGraphState) -> dict:
    """
    LangGraph node — decompose state["question"] into sub-questions.

    Returns a partial state update with:
      sub_questions, iterations, execution_mode, cypher_sufficient, hop_dependencies
    """
    cfg      = get_config()
    question = state["question"]
    max_q: int = cfg["agentic_rag"]["max_sub_questions"]

    try:
        raw  = _call(question, cfg)
        subs, exec_mode, cypher_ok, deps = _parse_json(raw, question, max_q)
    except Exception as exc:
        log.error("Decomposer failed — using original question as single parallel hop: %s", exc)
        subs      = [question]
        exec_mode = "parallel"
        cypher_ok = False
        deps      = [None]

    log.info(
        "Decomposed into %d sub-question(s) | mode=%s | cypher_sufficient=%s | deps=%s",
        len(subs), exec_mode, cypher_ok, deps,
    )
    for i, (q, d) in enumerate(zip(subs, deps)):
        log.debug("  hop %d (depends_on=%s): %s", i, d, q)

    return {
        "sub_questions":    subs,
        "iterations":       state.get("iterations", 0) + 1,
        "execution_mode":   exec_mode,
        "cypher_sufficient": cypher_ok,
        "hop_dependencies": deps,
    }
