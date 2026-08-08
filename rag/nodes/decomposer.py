"""
rag/nodes/decomposer.py

LangGraph node: decompose the user question into at most max_sub_questions
atomic sub-questions using Claude Haiku at temperature=0.

Simple self-contained questions are returned unchanged as a single-item list.
Falls back to [original_question] on any LLM failure so the pipeline never stalls.
"""

from __future__ import annotations

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
        max_tokens  = 256,
        temperature = cfg["llm"]["chat_temperature"],
        messages    = [{"role": "user", "content": _PROMPT.format(
            max_q=max_q, question=question,
        )}],
    )
    return msg.content[0].text.strip()


def decompose(state: FinGraphState) -> dict:
    """
    LangGraph node — decompose state["question"] into sub-questions.
    Returns: {"sub_questions": [...], "iterations": n+1}
    """
    cfg = get_config()
    question = state["question"]
    max_q: int = cfg["agentic_rag"]["max_sub_questions"]

    try:
        raw   = _call(question, cfg)
        items = re.findall(r"^\d+\.\s*(.+)$", raw, re.MULTILINE)
        subs  = [q.strip() for q in items if q.strip()][:max_q] or [question]
    except Exception as exc:
        log.error("Decomposer failed -- using original question: %s", exc)
        subs = [question]

    log.info("Decomposed into %d sub-question(s): %s", len(subs), subs)
    return {"sub_questions": subs, "iterations": state.get("iterations", 0) + 1}
