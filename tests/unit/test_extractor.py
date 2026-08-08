"""
tests/unit/test_extractor.py

Unit tests for ingestion/extractor.py.
No Anthropic API calls — all Haiku interactions are mocked.

Covers:
  - Entity type whitelist completeness
  - Prompt rendering (including curly-brace edge case)
  - JSON parsing: clean / markdown-fenced / invalid type / missing fields / blank strings
  - Case-insensitive entity type normalisation
  - _enrich(): empty body fast-path, known-type preservation, unknown-type override
  - _enrich(): error handling (extraction_error key, zero tokens)
  - _enrich(): token count propagation
  - Resume logic (done-prid skipping)
  - Pydantic model validation

Run from rbi_intel/:
    pytest tests/unit/test_extractor.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.extractor import (
    _PROMPT_STATIC,
    _TYPE_LOWER,
    _VALID_TYPES,
    _build_prompt,
    _enrich,
)
from ingestion.validator import EnforcementExtractionOutput


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> dict[str, Any]:
    """Minimal config stub — no disk read required."""
    return {
        "models": {"system_model": "claude-haiku-4-5-20251001"},
        "llm": {"max_tokens_extraction": 1024, "extraction_temperature": 0},
    }


@pytest.fixture
def known_type_record() -> dict[str, Any]:
    """Record whose entity_type was already resolved by the scraper."""
    return {
        "prid": 99001,
        "entity_name": "Kotak Mahindra Bank Ltd",
        "entity_type": "Bank-Private",
        "body_text": (
            "RBI imposed a monetary penalty of Rs 2 crore on Kotak Mahindra Bank "
            "for non-compliance with KYC directions under Section 47A(1)(c) of the "
            "Banking Regulation Act 1949."
        ),
    }


@pytest.fixture
def unknown_type_record() -> dict[str, Any]:
    """Record whose entity_type the scraper could not determine."""
    return {
        "prid": 99002,
        "entity_name": "Janakalyan Sahakari Bank Ltd",
        "entity_type": None,
        "body_text": (
            "RBI has cancelled the Certificate of Registration of Janakalyan "
            "Sahakari Bank Ltd, an urban co-operative bank, for failing to comply "
            "with minimum capital requirements under Section 56 of the Banking "
            "Regulation Act 1949."
        ),
    }


@pytest.fixture
def haiku_output_factory():
    """Returns an EnforcementExtractionOutput with given values."""
    def _make(
        entity_type: str = "Bank-UCB",
        violations: list[str] | None = None,
        regulation_refs: list[str] | None = None,
    ) -> EnforcementExtractionOutput:
        return EnforcementExtractionOutput(
            entity_type=entity_type,
            violations=violations or ["failure to maintain minimum capital requirements"],
            regulation_refs=regulation_refs or ["Section 56 of Banking Regulation Act 1949"],
        )
    return _make


# ── 1. Entity type whitelist ───────────────────────────────────────────────────

def test_valid_types_contains_all_18():
    expected = {
        "Bank-Private", "Bank-Public", "Bank-UCB", "Bank-Cooperative",
        "Bank-DCCB", "Bank-SCB", "Bank-RRB", "Bank-SFB", "Bank-Payment",
        "NBFC-P2P", "NBFC-ICC", "NBFC-HFC", "NBFC-MFI",
        "ARC", "CIC", "PA", "PPI", "UNKNOWN",
    }
    assert _VALID_TYPES == expected, f"Missing: {expected - _VALID_TYPES}"


def test_type_lower_covers_all_valid_types():
    """Every canonical type must be reachable via lowercase lookup."""
    for t in _VALID_TYPES:
        assert _TYPE_LOWER[t.lower()] == t


# ── 2. Prompt rendering ────────────────────────────────────────────────────────

def test_prompt_contains_entity_name():
    prompt = _build_prompt("HDFC Bank Ltd", "Bank-Private", "some body")
    assert "HDFC Bank Ltd" in prompt


def test_prompt_contains_entity_type():
    prompt = _build_prompt("X Bank", "Bank-UCB", "some body")
    assert "Bank-UCB" in prompt


def test_prompt_contains_body_text():
    body = "penalty imposed for non-compliance with KYC norms"
    prompt = _build_prompt("X", "UNKNOWN", body)
    assert body in prompt


def test_prompt_body_with_curly_braces_no_keyerror():
    """
    body_text containing { or } must NOT raise KeyError.
    This guards against statute refs like "Section 47A{1}(c)" or
    encoding artefacts that embed brace characters.
    """
    body = "penalty under Section 47A{1}(c) — refer {circular} for details"
    prompt = _build_prompt("Test Bank", "UNKNOWN", body)  # must not raise
    assert "{1}" in prompt          # braces preserved literally
    assert "{circular}" in prompt   # not interpolated away


def test_prompt_contains_json_skeleton():
    """The JSON output template must be present in the static constant and rendered prompt."""
    assert '"entity_type"' in _PROMPT_STATIC
    assert '"violations"' in _PROMPT_STATIC
    assert '"regulation_refs"' in _PROMPT_STATIC
    prompt = _build_prompt("X", "UNKNOWN", "body")
    assert '"entity_type"' in prompt
    assert '"violations"' in prompt
    assert '"regulation_refs"' in prompt


def test_prompt_unknown_string_not_none():
    """When entity_type is coerced to 'UNKNOWN', the literal string 'None' must not appear."""
    prompt = _build_prompt("Some Bank", "UNKNOWN", "body text")
    assert "None" not in prompt


# ── 3. JSON parsing (isolated, no mock needed) ─────────────────────────────────

def _simulate_parse(raw: str) -> EnforcementExtractionOutput:
    """Replicate _call_haiku's parsing logic for isolated unit testing."""
    if raw.startswith("```"):
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("```"))
    parsed: dict[str, Any] = json.loads(raw)
    etype: str = str(parsed.get("entity_type", "UNKNOWN")).strip()
    etype = _TYPE_LOWER.get(etype.lower(), "UNKNOWN")
    return EnforcementExtractionOutput(
        entity_type=etype,
        violations=[str(v) for v in parsed.get("violations", []) if str(v).strip()],
        regulation_refs=[str(r) for r in parsed.get("regulation_refs", []) if str(r).strip()],
    )


def test_parse_clean_json():
    raw = json.dumps({
        "entity_type": "NBFC-ICC",
        "violations": ["non-compliance with fair practices code"],
        "regulation_refs": ["Section 45-IA of RBI Act 1934"],
    })
    out = _simulate_parse(raw)
    assert out.entity_type == "NBFC-ICC"
    assert len(out.violations) == 1
    assert len(out.regulation_refs) == 1


def test_parse_markdown_fenced_json():
    raw = '```json\n{"entity_type": "Bank-UCB", "violations": [], "regulation_refs": []}\n```'
    out = _simulate_parse(raw)
    assert out.entity_type == "Bank-UCB"


def test_parse_markdown_fence_without_language_tag():
    raw = '```\n{"entity_type": "PA", "violations": [], "regulation_refs": []}\n```'
    out = _simulate_parse(raw)
    assert out.entity_type == "PA"


def test_parse_case_insensitive_entity_type():
    """Haiku returning wrong case must still resolve to canonical type."""
    for haiku_response, expected in [
        ("bank-private", "Bank-Private"),
        ("NBFC-ICC", "NBFC-ICC"),
        ("nbfc-p2p", "NBFC-P2P"),
        ("arc", "ARC"),
        ("unknown", "UNKNOWN"),
    ]:
        raw = json.dumps({"entity_type": haiku_response, "violations": [], "regulation_refs": []})
        out = _simulate_parse(raw)
        assert out.entity_type == expected, f"Failed for '{haiku_response}'"


def test_parse_invalid_entity_type_becomes_unknown():
    raw = json.dumps({"entity_type": "Commercial_Bank", "violations": [], "regulation_refs": []})
    out = _simulate_parse(raw)
    assert out.entity_type == "UNKNOWN"


def test_parse_missing_violations_defaults_to_empty_list():
    raw = json.dumps({"entity_type": "PA", "regulation_refs": ["Section 18 of PSS Act 2007"]})
    out = _simulate_parse(raw)
    assert out.violations == []


def test_parse_missing_regulation_refs_defaults_to_empty_list():
    raw = json.dumps({"entity_type": "NBFC-MFI", "violations": ["non-compliance with MFIN code"]})
    out = _simulate_parse(raw)
    assert out.regulation_refs == []


def test_parse_blank_string_items_are_filtered():
    """Empty / whitespace-only strings in lists must be dropped."""
    raw = json.dumps({
        "entity_type": "ARC",
        "violations": ["valid violation", "", "  "],
        "regulation_refs": ["Section 9 of SARFAESI Act 2002", ""],
    })
    out = _simulate_parse(raw)
    assert out.violations == ["valid violation"]
    assert out.regulation_refs == ["Section 9 of SARFAESI Act 2002"]


# ── 4. _enrich() behaviour ─────────────────────────────────────────────────────

def test_enrich_empty_body_returns_early_without_api_call(cfg):
    """Empty body_text must short-circuit before calling Haiku."""
    record = {"prid": 1, "entity_name": "X", "entity_type": "Bank-UCB", "body_text": ""}
    with patch("ingestion.extractor._call_haiku") as mock_haiku:
        result = _enrich(record, client=MagicMock(), cfg=cfg)
    mock_haiku.assert_not_called()
    assert result["violations"] == []
    assert result["regulation_refs"] == []
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0


def test_enrich_whitespace_only_body_returns_early(cfg):
    record = {"prid": 2, "entity_name": "Y", "entity_type": "NBFC-ICC", "body_text": "   \n  "}
    with patch("ingestion.extractor._call_haiku") as mock_haiku:
        _enrich(record, client=MagicMock(), cfg=cfg)
    mock_haiku.assert_not_called()


def test_enrich_known_type_is_never_overridden(known_type_record, cfg, haiku_output_factory):
    """Haiku classifies as NBFC-ICC but original Bank-Private must be preserved."""
    haiku_out = haiku_output_factory(entity_type="NBFC-ICC")
    with patch("ingestion.extractor._call_haiku", return_value=(haiku_out, 300, 80)):
        result = _enrich(known_type_record, client=MagicMock(), cfg=cfg)
    assert result["entity_type"] == "Bank-Private"


def test_enrich_unknown_type_is_resolved_by_haiku(unknown_type_record, cfg, haiku_output_factory):
    """None entity_type must be replaced with Haiku's classification."""
    haiku_out = haiku_output_factory(entity_type="Bank-UCB")
    with patch("ingestion.extractor._call_haiku", return_value=(haiku_out, 280, 70)):
        result = _enrich(unknown_type_record, client=MagicMock(), cfg=cfg)
    assert result["entity_type"] == "Bank-UCB"


def test_enrich_violations_and_refs_are_set(known_type_record, cfg, haiku_output_factory):
    haiku_out = haiku_output_factory(
        violations=["non-compliance with KYC norms", "failure to file SAR within stipulated time"],
        regulation_refs=["Section 47A(1)(c) of Banking Regulation Act 1949"],
    )
    with patch("ingestion.extractor._call_haiku", return_value=(haiku_out, 500, 110)):
        result = _enrich(known_type_record, client=MagicMock(), cfg=cfg)
    assert len(result["violations"]) == 2
    assert len(result["regulation_refs"]) == 1


def test_enrich_tokens_propagated_to_result(known_type_record, cfg, haiku_output_factory):
    haiku_out = haiku_output_factory()
    with patch("ingestion.extractor._call_haiku", return_value=(haiku_out, 750, 120)):
        result = _enrich(known_type_record, client=MagicMock(), cfg=cfg)
    assert result["input_tokens"] == 750
    assert result["output_tokens"] == 120


def test_enrich_api_error_sets_extraction_error_key(known_type_record, cfg):
    """On all-retries-failed, record gets extraction_error key and zero tokens."""
    with patch("ingestion.extractor._call_haiku", side_effect=RuntimeError("503 overloaded")):
        result = _enrich(known_type_record, client=MagicMock(), cfg=cfg)
    assert "extraction_error" in result
    assert "503 overloaded" in result["extraction_error"]
    assert result["violations"] == []
    assert result["regulation_refs"] == []
    assert result["input_tokens"] == 0


def test_enrich_api_error_original_fields_preserved(known_type_record, cfg):
    """On error, the original scraper fields (prid, entity_name, etc.) must survive."""
    with patch("ingestion.extractor._call_haiku", side_effect=RuntimeError("timeout")):
        result = _enrich(known_type_record, client=MagicMock(), cfg=cfg)
    assert result["prid"] == 99001
    assert result["entity_name"] == "Kotak Mahindra Bank Ltd"


# ── 5. Resume logic ────────────────────────────────────────────────────────────

def test_resume_skips_already_processed_prids():
    all_records = [{"prid": i} for i in range(1, 6)]   # prids 1–5
    done_prids  = {r["prid"] for r in [{"prid": 1}, {"prid": 2}, {"prid": 3}]}
    todo = [r for r in all_records if r["prid"] not in done_prids]
    assert [r["prid"] for r in todo] == [4, 5]


def test_resume_with_empty_done_set_processes_all():
    all_records = [{"prid": i} for i in range(1, 4)]
    done_prids: set[int] = set()
    todo = [r for r in all_records if r["prid"] not in done_prids]
    assert len(todo) == 3


# ── 6. Pydantic model ──────────────────────────────────────────────────────────

def test_enforcement_extraction_output_valid():
    obj = EnforcementExtractionOutput(
        entity_type="Bank-UCB",
        violations=["non-compliance with KYC norms"],
        regulation_refs=["Section 47A(1)(c) of Banking Regulation Act 1949"],
    )
    assert obj.entity_type == "Bank-UCB"
    assert len(obj.violations) == 1


def test_enforcement_extraction_output_defaults_to_empty_lists():
    obj = EnforcementExtractionOutput(entity_type="UNKNOWN")
    assert obj.violations == []
    assert obj.regulation_refs == []
