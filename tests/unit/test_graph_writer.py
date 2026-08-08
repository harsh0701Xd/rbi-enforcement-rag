"""
tests/unit/test_graph_writer.py

Unit tests for ingestion/graph_writer.py (build_rows + _norm).
No Neo4j connection — pure logic tests.

Covers:
  - _norm(): lowercases and strips whitespace
  - build_rows(): correct row structure for all 7 batch types
  - build_rows(): Institution deduplication by name
  - build_rows(): Violation global deduplication by text_key
  - build_rows(): Regulation global deduplication by citation_key
  - build_rows(): empty violations / regulation_refs produce no rows
  - build_rows(): violation text preserved on first encounter
  - build_rows(): PenaltyAction row carries all required Neo4j properties

Run from rbi_intel/:
    pytest tests/unit/test_graph_writer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.graph_writer import _norm, build_rows


# ── Fixture helper ────────────────────────────────────────────────────────────

def _record(
    prid=1,
    entity_name="Test Bank Ltd",
    entity_type="Bank-Private",
    penalty_lakh=100.0,
    action_type="monetary_penalty",
    violations=None,
    regulation_refs=None,
):
    return {
        "prid": prid,
        "date": "2023-04-15",
        "title": f"RBI imposes penalty on {entity_name}",
        "entity_name": entity_name,
        "entity_type": entity_type,
        "penalty_lakh": penalty_lakh,
        "action_type": action_type,
        "source_url": f"https://rbi.org.in/pr/{prid}",
        "pr_number": f"2023/{prid}",
        "body_text": "Some body text.",
        "violations": violations if violations is not None else ["Non-compliance with KYC directions"],
        "regulation_refs": regulation_refs if regulation_refs is not None else ["Section 47A(1)(c) of Banking Regulation Act 1949"],
        "is_bulk_cancellation": False,
        "input_tokens": 500,
        "output_tokens": 80,
    }


# ── 1. _norm ──────────────────────────────────────────────────────────────────

def test_norm_lowercases():
    assert _norm("Non-compliance with KYC Directions") == "non-compliance with kyc directions"

def test_norm_strips_whitespace():
    assert _norm("  Section 47A  ") == "section 47a"

def test_norm_already_normalised():
    assert _norm("section 47a") == "section 47a"


# ── 2. build_rows: basic structure ────────────────────────────────────────────

def test_build_rows_returns_all_seven_keys():
    rows = build_rows([_record()])
    assert set(rows.keys()) == {
        "institutions", "penalty_actions", "violations",
        "regulations", "received", "for_violation", "cites",
    }

def test_build_rows_single_record_produces_one_institution():
    rows = build_rows([_record()])
    assert len(rows["institutions"]) == 1
    assert rows["institutions"][0]["name"] == "Test Bank Ltd"
    assert rows["institutions"][0]["entity_type"] == "Bank-Private"

def test_build_rows_single_record_produces_one_penalty_action():
    rows = build_rows([_record(prid=42)])
    assert len(rows["penalty_actions"]) == 1
    pa = rows["penalty_actions"][0]
    assert pa["prid"] == 42
    assert pa["date"] == "2023-04-15"
    assert pa["penalty_lakh"] == 100.0
    assert pa["action_type"] == "monetary_penalty"
    assert "source_url" in pa
    assert "pr_number" in pa

def test_build_rows_single_record_produces_received_rel():
    rows = build_rows([_record(prid=7)])
    assert len(rows["received"]) == 1
    assert rows["received"][0] == {"institution_name": "Test Bank Ltd", "prid": 7}


# ── 3. Institution deduplication ──────────────────────────────────────────────

def test_institution_deduplicated_across_two_penalties():
    records = [
        _record(prid=1, entity_name="HDFC Bank Ltd"),
        _record(prid=2, entity_name="HDFC Bank Ltd"),
    ]
    rows = build_rows(records)
    assert len(rows["institutions"]) == 1
    assert len(rows["penalty_actions"]) == 2
    assert len(rows["received"]) == 2

def test_two_distinct_institutions_produce_two_rows():
    records = [
        _record(prid=1, entity_name="HDFC Bank Ltd"),
        _record(prid=2, entity_name="ICICI Bank Ltd"),
    ]
    rows = build_rows(records)
    assert len(rows["institutions"]) == 2


# ── 4. Violation global deduplication ────────────────────────────────────────

def test_same_violation_text_deduplicates_globally():
    shared = "Non-compliance with KYC directions"
    records = [
        _record(prid=1, entity_name="Bank A", violations=[shared]),
        _record(prid=2, entity_name="Bank B", violations=[shared]),
    ]
    rows = build_rows(records)
    assert len(rows["violations"]) == 1
    assert len(rows["for_violation"]) == 2

def test_violation_text_key_is_normalised():
    records = [
        _record(prid=1, violations=["  Non-compliance with KYC Directions  "]),
        _record(prid=2, violations=["non-compliance with kyc directions"]),
    ]
    rows = build_rows(records)
    assert len(rows["violations"]) == 1

def test_violation_original_text_preserved():
    original = "Non-compliance with KYC Directions"
    rows = build_rows([_record(violations=[original])])
    assert rows["violations"][0]["text"] == original
    assert rows["violations"][0]["text_key"] == _norm(original)

def test_multiple_distinct_violations_produce_multiple_nodes():
    rows = build_rows([_record(violations=["Violation A", "Violation B", "Violation C"])])
    assert len(rows["violations"]) == 3
    assert len(rows["for_violation"]) == 3


# ── 5. Regulation global deduplication ───────────────────────────────────────

def test_same_regulation_deduplicates_globally():
    shared = "Section 47A(1)(c) of Banking Regulation Act 1949"
    records = [
        _record(prid=1, entity_name="Bank A", regulation_refs=[shared]),
        _record(prid=2, entity_name="Bank B", regulation_refs=[shared]),
    ]
    rows = build_rows(records)
    assert len(rows["regulations"]) == 1
    assert len(rows["cites"]) == 2

def test_regulation_citation_key_is_normalised():
    records = [
        _record(prid=1, regulation_refs=["  Section 47A  "]),
        _record(prid=2, regulation_refs=["section 47a"]),
    ]
    rows = build_rows(records)
    assert len(rows["regulations"]) == 1


# ── 6. Empty violations / regulation_refs ────────────────────────────────────

def test_empty_violations_produces_no_violation_rows():
    rows = build_rows([_record(violations=[], regulation_refs=["Some Act"])])
    assert rows["violations"] == []
    assert rows["for_violation"] == []

def test_empty_regulation_refs_produces_no_regulation_rows():
    rows = build_rows([_record(violations=["Some violation"], regulation_refs=[])])
    assert rows["regulations"] == []
    assert rows["cites"] == []


# ── 7. Multiple records end-to-end count check ───────────────────────────────

def test_three_records_with_shared_violation_and_regulation():
    shared_v = "KYC non-compliance"
    shared_r = "Section 47A BR Act 1949"
    records = [
        _record(prid=i, entity_name=f"Bank {i}", violations=[shared_v], regulation_refs=[shared_r])
        for i in range(1, 4)
    ]
    rows = build_rows(records)
    assert len(rows["institutions"]) == 3
    assert len(rows["penalty_actions"]) == 3
    assert len(rows["violations"]) == 1
    assert len(rows["regulations"]) == 1
    assert len(rows["received"]) == 3
    assert len(rows["for_violation"]) == 3
    assert len(rows["cites"]) == 3
