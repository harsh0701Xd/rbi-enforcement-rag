"""
rag/_synthesis.py

Shared answer-synthesis function used by all three RAG stages.

Eliminates the duplicate _synthesise() / _call_haiku() definitions that
previously existed in naive_rag.py, advanced_rag.py, and nodes/synthesiser.py.

Public API:
    get_client()                            -- shared Anthropic singleton
    synthesise(context, question, cfg, system,
               *, return_usage=False)       -- retrying LLM call; raises on hard failure
                                               pass return_usage=True to get
                                               (answer_str, usage_dict) tuple
"""

from __future__ import annotations

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from prompts.synthesis import USER_TEMPLATE

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    """Return the module-level Anthropic singleton, creating it on first call."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    reraise=True,
)
def synthesise(
    context: str,
    question: str,
    cfg: dict,
    system: str,
    *,
    return_usage: bool = False,
) -> "str | tuple[str, dict]":
    """
    Call Claude to synthesise an answer from retrieved context.

    Args:
        context:      Pre-formatted context string (vector chunks and/or graph rows).
        question:     Original user question.
        cfg:          Config dict (uses models.system_model, llm.max_tokens_chat,
                      llm.chat_temperature).
        system:       System prompt -- pass prompts.synthesis.SYSTEM for Stages 1/2,
                      or prompts.synthesis.AGENTIC_SYSTEM for Stage 3.
        return_usage: If True, return (answer_text, usage_dict) instead of the
                      answer string alone.  usage_dict has keys "input_tokens" and
                      "output_tokens".  Used by inference logging; default False
                      preserves existing call sites unchanged.

    Returns:
        Stripped answer string, or (answer_str, usage_dict) when return_usage=True.

    Raises:
        anthropic.RateLimitError after 3 retries (caller should catch and degrade).
    """
    msg = get_client().messages.create(
        model       = cfg["models"]["system_model"],
        max_tokens  = cfg["llm"]["max_tokens_chat"],
        temperature = cfg["llm"]["chat_temperature"],
        system      = system,
        messages    = [{"role": "user", "content": USER_TEMPLATE.format(
            context=context, question=question,
        )}],
    )
    text = msg.content[0].text.strip()
    if return_usage:
        usage = {
            "input_tokens":  msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        return text, usage
    return text
