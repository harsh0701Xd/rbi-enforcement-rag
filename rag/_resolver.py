"""
rag/_resolver.py

Template variable resolution for Stage 6 multi-hop dependent reasoning.

Stage 6 introduces open-field placeholder syntax:
    {q0.entity_name}   →  hop_results[0]["entity_name"]
    {q0.year}          →  hop_results[0]["year"]
    {q0.penalty_lakh}  →  hop_results[0]["penalty_lakh"]

The field name is looked up directly on the raw rows[0] dict stored in
hop_results.  Alias-prefixed keys (e.g. "i.entity_name") are also matched
by stripping the prefix.

Legacy syntax ({q0_answer}) is still supported for backward compatibility —
it extracts the "best" string value from the rows[0] dict using a standard
priority list.

Public API:
    resolve_template(q, hop_results)  -> str   substitute {q0.field} and {q0_answer}
    has_template_vars(q)              -> bool   detect unresolved template variables
"""

from __future__ import annotations

import re

# Stage 6: {q0.field_name} — open-field column lookup on raw rows[0]
_FIELD_RE = re.compile(r"\{q(\d+)\.(\w+)\}")

# Legacy Stage 5: {q0_answer} — retained for backward compatibility
_LEGACY_RE = re.compile(r"\{q(\d+)_answer\}")

# Priority order for legacy extraction from a raw row dict
_PRIORITY_KEYS = ("entity_name", "i.name", "name", "institution_name")


def _extract_string(row: dict, field: str | None = None) -> str | None:
    """
    Extract a string value from a raw rows[0] dict.

    If *field* is given, try exact key match then alias-stripped match.
    Fall back to priority list then first non-empty string value.
    """
    if field is not None:
        # Exact match
        val = row.get(field)
        if val is not None:
            return str(val)
        # Alias-stripped match: "i.entity_name" → field="entity_name"
        for k, v in row.items():
            if k.split(".")[-1] == field and v is not None:
                return str(v)
    # Priority-list fallback (used by legacy {q0_answer} and field-miss cases)
    for key in _PRIORITY_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Last resort: first non-empty string value in the row
    for v in row.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def resolve_template(q: str, hop_results: dict[int, "dict | str"]) -> str:
    """
    Replace {qN.field} and {qN_answer} placeholders in *q* with resolved values.

    hop_results maps hop_id (int) to either:
      - a raw rows[0] dict  (Stage 6 — new)
      - a string            (Stage 5 legacy — backward compat)

    Unresolvable placeholders (hop not completed, field missing) are left as-is
    so callers can detect and log the failure rather than silently producing
    wrong output.

    Examples:
        >>> resolve_template("Violations for {q0.entity_name}?", {0: {"entity_name": "Axis Bank"}})
        'Violations for Axis Bank?'
        >>> resolve_template("Penalties in {q0.year}?", {0: {"year": 2022}})
        'Penalties in 2022?'
        >>> resolve_template("What did {q0_answer} do?", {0: "Axis Bank"})   # legacy
        'What did Axis Bank do?'
        >>> resolve_template("Violations for {q0.entity_name}?", {})         # missing hop
        'Violations for {q0.entity_name}?'
    """
    def _replace_field(m: re.Match) -> str:
        hop_id = int(m.group(1))
        field  = m.group(2)
        hop_val = hop_results.get(hop_id)
        if hop_val is None:
            return m.group(0)  # leave placeholder — hop not yet completed
        if isinstance(hop_val, dict):
            result = _extract_string(hop_val, field)
            return result if result is not None else m.group(0)
        # String (legacy): ignore field name, return value directly
        return str(hop_val)

    def _replace_legacy(m: re.Match) -> str:
        hop_id = int(m.group(1))
        hop_val = hop_results.get(hop_id)
        if hop_val is None:
            return m.group(0)
        if isinstance(hop_val, dict):
            result = _extract_string(hop_val)
            return result if result is not None else m.group(0)
        return str(hop_val)

    # Apply new syntax first, then legacy
    q = _FIELD_RE.sub(_replace_field, q)
    q = _LEGACY_RE.sub(_replace_legacy, q)
    return q


def has_template_vars(q: str) -> bool:
    """
    Return True if *q* contains any unresolved template variables.

    Detects both Stage 6 ({q0.field}) and legacy Stage 5 ({q0_answer}) syntax.

    Examples:
        >>> has_template_vars("Violations for {q0.entity_name}?")
        True
        >>> has_template_vars("Violations for {q0_answer}?")
        True
        >>> has_template_vars("What penalty did Axis Bank receive?")
        False
    """
    return bool(_FIELD_RE.search(q) or _LEGACY_RE.search(q))
