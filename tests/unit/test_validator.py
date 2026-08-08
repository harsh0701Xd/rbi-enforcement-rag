"""
tests/unit/test_validator.py

Unit tests for ingestion/validator.py (EnforcementRecord model + helpers).
No disk I/O — all record dicts are constructed inline.

Covers:
  - _derive_action_type(): monetary_penalty / cor_cancellation / other
  - _derive_is_bulk(): cor_cancellation flag + entity_name == title fallback
  - _to_record(): field mapping, entity_type coercion, pr_number str conversion
  - EnforcementRecord Pydantic model: valid construction, defaults
  - Edge cases: None entity_type, missing optional fields

Run from rbi_intel/:
    pytest tests/unit/test_validator.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.validator import EnforcementRecord, _derive_action_type, _derive_is_bulk, _to_record


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _base_raw() -> dict:
    """Minimal valid raw record with all required fields."""
    return {
        "prid":           99001,
        "date":           "2023-04-15",
        "title":          "RBI imposes monetary penalty on Test Bank Ltd",
        "entity_name":    "Test Bank Ltd",
        "entity_type":    "Bank-Private",
        "penalty_lakh":   100.0,
        "other_action":   None,
        "source_url":     "https://rbi.org.in/pr/99001",
        "pr_number":      "2023-2024/99",
        "body_text":      "RBI imposed a penalty of Rs 1 crore on Test Bank Ltd.",
        "violations":     ["Non-compliance with KYC directions"],
        "regulation_refs": ["Section 47A(1)(c) of Banking Regulation Act 1949"],
        "input_tokens":   500,
        "output_tokens":  80,
    }


def _cor_raw() -> dict:
    """Raw record representing a bulk CoR cancellation."""
    return {
        "prid":           99002,
        "date":           "2023-01-10",
        "title":          "RBI cancels Certificate of Registration of Five NBFCs",
        "entity_name":    "RBI cancels Certificate of Registration of Five NBFCs",
        "entity_type":    "NBFC-ICC",
        "penalty_lakh":   None,
        "other_action":   "cor_cancellation",
        "source_url":     "https://rbi.org.in/pr/99002",
        "pr_number":      "2023-2024/5",
        "body_text":      "RBI has cancelled CoR of the following companies.",
        "violations":     [],
        "regulation_refs": ["Section 45-IA (6) of the Reserve Bank of India Act, 1934"],
        "input_tokens":   400,
        "output_tokens":  60,
    }


# ── 1. _derive_action_type ────────────────────────────────────────────────────

def test_action_type_monetary_penalty():
    raw = _base_raw()
    assert _derive_action_type(raw) == "monetary_penalty"


def test_action_type_cor_cancellation():
    raw = _cor_raw()
    assert _derive_action_type(raw) == "cor_cancellation"


def test_action_type_other_when_no_penalty_and_no_cor():
    raw = _base_raw()
    raw["penalty_lakh"] = None
    raw["other_action"] = None
    assert _derive_action_type(raw) == "other"


def test_action_type_cor_takes_precedence_over_penalty_lakh():
    """If somehow both fields are set, cor_cancellation wins."""
    raw = _base_raw()
    raw["other_action"] = "cor_cancellation"
    raw["penalty_lakh"] = 50.0
    assert _derive_action_type(raw) == "cor_cancellation"


# ── 2. _derive_is_bulk ────────────────────────────────────────────────────────

def test_is_bulk_true_when_other_action_is_cor():
    assert _derive_is_bulk(_cor_raw()) is True


def test_is_bulk_true_when_entity_name_equals_title():
    raw = _base_raw()
    raw["entity_name"] = raw["title"]   # headline used as entity_name
    assert _derive_is_bulk(raw) is True


def test_is_bulk_false_for_normal_monetary_penalty():
    assert _derive_is_bulk(_base_raw()) is False


def test_is_bulk_false_when_entity_name_differs_from_title():
    raw = _base_raw()
    raw["entity_name"] = "Karnataka Bank Limited"
    assert _derive_is_bulk(raw) is False


# ── 3. _to_record ─────────────────────────────────────────────────────────────

def test_to_record_maps_all_fields():
    record = _to_record(_base_raw())
    assert record.prid == 99001
    assert record.date == "2023-04-15"
    assert record.entity_name == "Test Bank Ltd"
    assert record.entity_type == "Bank-Private"
    assert record.penalty_lakh == 100.0
    assert record.action_type == "monetary_penalty"
    assert record.is_bulk_cancellation is False
    assert len(record.violations) == 1
    assert len(record.regulation_refs) == 1
    assert record.input_tokens == 500
    assert record.output_tokens == 80


def test_to_record_none_entity_type_coerced_to_unknown():
    raw = _base_raw()
    raw["entity_type"] = None
    record = _to_record(raw)
    assert record.entity_type == "UNKNOWN"


def test_to_record_empty_entity_type_coerced_to_unknown():
    raw = _base_raw()
    raw["entity_type"] = ""
    record = _to_record(raw)
    assert record.entity_type == "UNKNOWN"


def test_to_record_pr_number_coerced_to_string():
    raw = _base_raw()
    raw["pr_number"] = 12345   # integer from JSON
    record = _to_record(raw)
    assert isinstance(record.pr_number, str)
    assert record.pr_number == "12345"


def test_to_record_cor_sets_is_bulk_true():
    record = _to_record(_cor_raw())
    assert record.is_bulk_cancellation is True
    assert record.action_type == "cor_cancellation"
    assert record.penalty_lakh is None


def test_to_record_missing_violations_defaults_to_empty_list():
    raw = _base_raw()
    del raw["violations"]
    record = _to_record(raw)
    assert record.violations == []


def test_to_record_missing_regulation_refs_defaults_to_empty_list():
    raw = _base_raw()
    del raw["regulation_refs"]
    record = _to_record(raw)
    assert record.regulation_refs == []


# ── 4. EnforcementRecord Pydantic model ───────────────────────────────────────

def test_enforcement_record_valid_construction():
    record = EnforcementRecord(
        prid=1,
        date="2023-01-01",
        title="Test",
        entity_name="Test Bank",
        entity_type="Bank-Private",
        action_type="monetary_penalty",
        source_url="https://rbi.org.in/pr/1",
        pr_number="2023/1",
        body_text="Some body text.",
        penalty_lakh=50.0,
    )
    assert record.prid == 1
    assert record.violations == []
    assert record.regulation_refs == []
    assert record.is_bulk_cancellation is False
    assert record.input_tokens == 0
    assert record.output_tokens == 0


def test_enforcement_record_penalty_lakh_optional():
    record = EnforcementRecord(
        prid=2,
        date="2023-01-01",
        title="CoR cancellation",
        entity_name="Some NBFC",
        entity_type="NBFC-ICC",
        action_type="cor_cancellation",
        source_url="https://rbi.org.in/pr/2",
        pr_number="2023/2",
        body_text="CoR cancelled.",
    )
    assert record.penalty_lakh is None
