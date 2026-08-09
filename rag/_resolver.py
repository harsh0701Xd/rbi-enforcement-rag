"""
rag/_resolver.py

Template variable resolution utility for Stage 5 multi-hop dependent reasoning.

When the decomposer emits a dependent sub-question like:
    "What were the violations committed by {q0_answer}?"

the hop_executor calls resolve_template() after hop 0 completes to substitute
the actual entity name before extract_entities() and vector retrieval run.

This is the fix for the Stage 4 parallel-execution limitation: entity pre-filter
now receives the correct, concrete institution name for dependent hops.

Public API:
    resolve_template(q, hop_results)  -> str   substitute {q0_answer}, {q1_answer}, ...
    has_template_vars(q)              -> bool   detect unresolved template variables
"""

from __future__ import annotations

import re

# Matches {q0_answer}, {q1_answer}, {q2_answer}, ... (case-sensitive)
_TEMPLATE_VAR_RE = re.compile(r"\{q(\d+)_answer\}")


def resolve_template(q: str, hop_results: dict[int, str]) -> str:
    """
    Replace {qN_answer} placeholders in *q* with the resolved answer from hop N.

    Args:
        q:           Sub-question string that may contain template variables.
        hop_results: Mapping of hop_id (int) -> resolved answer string,
                     built incrementally by the hop executor as hops complete.

    Returns:
        The resolved sub-question string.  Unresolvable placeholders (hop not
        yet completed or missing from hop_results) are left as-is so callers
        can log a warning rather than silently producing wrong output.

    Examples:
        >>> resolve_template("Violations for {q0_answer}?", {0: "Axis Bank Limited"})
        'Violations for Axis Bank Limited?'
        >>> resolve_template("Penalties for {q0_answer} in {q1_answer}?", {0: "Axis Bank", 1: "2023"})
        'Penalties for Axis Bank in 2023?'
        >>> resolve_template("What did {q0_answer} do?", {})   # hop 0 not yet resolved
        'What did {q0_answer} do?'
    """
    def _replace(m: re.Match) -> str:
        hop_id = int(m.group(1))
        return hop_results.get(hop_id, m.group(0))  # leave placeholder if missing

    return _TEMPLATE_VAR_RE.sub(_replace, q)


def has_template_vars(q: str) -> bool:
    """
    Return True if *q* contains any unresolved {qN_answer} template variables.

    Used by the hop executor to detect dependent hops that cannot be run yet.

    Examples:
        >>> has_template_vars("Violations for {q0_answer}?")
        True
        >>> has_template_vars("What penalty did Axis Bank receive?")
        False
    """
    return bool(_TEMPLATE_VAR_RE.search(q))
