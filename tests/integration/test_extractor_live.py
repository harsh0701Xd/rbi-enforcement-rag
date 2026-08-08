"""
tests/integration/test_extractor_live.py

Live smoke test for ingestion/extractor.py.
Makes 3 real Haiku API calls against records from rbi_enforcement_raw.json.

What this verifies end-to-end:
  1. Prompt renders without error for real records
  2. Haiku returns parseable JSON (not prose, not markdown)
  3. entity_type is a canonical value (or UNKNOWN — never garbage)
  4. violations and regulation_refs are non-null lists
  5. Pydantic model accepts the extraction output

Prerequisites:
  - ANTHROPIC_API_KEY must be set in the environment (or .env loaded)
  - data/raw/rbi_enforcement_raw.json must exist

Cost: ~3 Haiku calls x ~1000 tokens input x $1/MTok = $0.003

Run ONLY after unit tests pass:
    pytest tests/integration/test_extractor_live.py -v -s
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env before pytestmark evaluates os.getenv() — otherwise the key is
# never in the environment at import time and all tests are skipped.
load_dotenv()

# ── Skip entire module if API key is absent ────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set and not found in .env — skipping live API tests",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_sample_records(n: int = 3) -> list[dict[str, Any]]:
    """
    Pick n diverse records from raw data:
      - one with known entity_type
      - one with UNKNOWN / None entity_type
      - one NBFC (different sector from the first two)
    Falls back to first n records if corpus doesn't have enough variety.
    """
    raw_path = Path(__file__).parent.parent.parent / "data" / "raw" / "rbi_enforcement_raw.json"
    if not raw_path.exists():
        pytest.skip(f"Raw data not found: {raw_path}")

    all_records: list[dict[str, Any]] = json.loads(
        raw_path.read_text(encoding="utf-8")
    )["records"]

    chosen: list[dict[str, Any]] = []

    # 1. Known bank type
    for r in all_records:
        if r.get("entity_type") and "Bank-" in str(r["entity_type"]) and r.get("body_text"):
            chosen.append(r)
            break

    # 2. Unknown / None entity_type
    for r in all_records:
        if not r.get("entity_type") and r.get("body_text") and r not in chosen:
            chosen.append(r)
            break

    # 3. NBFC
    for r in all_records:
        if r.get("entity_type") and "NBFC" in str(r["entity_type"]) and r not in chosen and r.get("body_text"):
            chosen.append(r)
            break

    # Fallback: just take the first n with body_text
    if len(chosen) < n:
        for r in all_records:
            if r.get("body_text") and r not in chosen:
                chosen.append(r)
            if len(chosen) == n:
                break

    return chosen[:n]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_live_extraction_returns_valid_structure():
    """
    Run _enrich() against 3 real records.
    Asserts output structure, canonical type, and non-null list fields.
    """
    from anthropic import Anthropic
    from ingestion.extractor import _enrich
    from ingestion.validator import EnforcementExtractionOutput
    from config import get_config

    cfg = get_config()
    client = Anthropic()
    records = _load_sample_records(3)

    assert len(records) >= 1, "No usable records found in raw data"

    from ingestion.extractor import _VALID_TYPES

    successes = 0
    for record in records:
        prid = record["prid"]
        result = _enrich(record, client=client, cfg=cfg)

        if "extraction_error" in result:
            # Print but don't immediately fail — one bad record (content policy edge
            # case, malformed PR body, etc.) must not abort the whole test.
            # The full-run error rate is what matters; check successes at the end.
            print(f"\n  prid={prid} WARN: extraction_error — {result['extraction_error']}")
            continue

        # entity_type must be a canonical string (never None after error-path coercion)
        assert result["entity_type"] in _VALID_TYPES, (
            f"prid={prid} bad entity_type: {result['entity_type']!r}"
        )

        # violations and regulation_refs must be lists (possibly empty)
        assert isinstance(result["violations"], list), f"prid={prid} violations not a list"
        assert isinstance(result["regulation_refs"], list), f"prid={prid} regulation_refs not a list"

        # All items in lists must be non-empty strings
        for v in result["violations"]:
            assert isinstance(v, str) and v.strip(), f"prid={prid} blank violation item"
        for r in result["regulation_refs"]:
            assert isinstance(r, str) and r.strip(), f"prid={prid} blank regulation_ref item"

        # Token counts must be positive for a successful API call
        assert result["input_tokens"] > 0, f"prid={prid} zero input_tokens"
        assert result["output_tokens"] > 0, f"prid={prid} zero output_tokens"

        # Pydantic model must accept the extraction output
        EnforcementExtractionOutput(
            entity_type=result["entity_type"],
            violations=result["violations"],
            regulation_refs=result["regulation_refs"],
        )

        print(f"\n  prid={prid}")
        print(f"    entity_name  : {result.get('entity_name')}")
        print(f"    entity_type  : {result['entity_type']}")
        print(f"    violations   : {result['violations'][:2]}")
        print(f"    refs         : {result['regulation_refs'][:2]}")
        print(f"    tokens in/out: {result['input_tokens']} / {result['output_tokens']}")
        successes += 1

    assert successes >= 1, (
        f"All {len(records)} records failed extraction — "
        "check ANTHROPIC_API_KEY, model name, and logs/errors.log"
    )


def test_live_unknown_entity_type_gets_resolved():
    """
    A record with entity_type=None must receive a non-UNKNOWN classification
    when enough context is present in body_text.
    (Test is advisory — UNKNOWN is acceptable if body is truly ambiguous.)
    """
    from anthropic import Anthropic
    from ingestion.extractor import _enrich, _VALID_TYPES
    from config import get_config

    cfg = get_config()
    client = Anthropic()

    raw_path = Path(__file__).parent.parent.parent / "data" / "raw" / "rbi_enforcement_raw.json"
    all_records = json.loads(raw_path.read_text(encoding="utf-8"))["records"]

    unknowns = [r for r in all_records if not r.get("entity_type") and r.get("body_text")]
    if not unknowns:
        pytest.skip("No UNKNOWN records in corpus")

    record = unknowns[0]
    result = _enrich(record, client=client, cfg=cfg)

    # After the error-path coercion fix, entity_type is always a valid string
    # (either Haiku's answer, or "UNKNOWN" on failure — never None).
    assert result["entity_type"] in _VALID_TYPES, (
        f"entity_type {result['entity_type']!r} not in valid set — coercion bug"
    )
    print(f"\n  prid={record['prid']} resolved to: {result['entity_type']}")
    if result["entity_type"] == "UNKNOWN":
        print("  (UNKNOWN is acceptable if body text is genuinely ambiguous)")


def test_live_known_type_not_overridden():
    """
    A record with entity_type=Bank-Private must retain Bank-Private after enrichment,
    regardless of what Haiku might infer independently.
    """
    from anthropic import Anthropic
    from ingestion.extractor import _enrich
    from config import get_config

    cfg = get_config()
    client = Anthropic()

    raw_path = Path(__file__).parent.parent.parent / "data" / "raw" / "rbi_enforcement_raw.json"
    all_records = json.loads(raw_path.read_text(encoding="utf-8"))["records"]

    banks = [r for r in all_records if r.get("entity_type") == "Bank-Private" and r.get("body_text")]
    if not banks:
        pytest.skip("No Bank-Private records in corpus")

    record = banks[0]
    result = _enrich(record, client=client, cfg=cfg)

    assert result["entity_type"] == "Bank-Private", (
        f"Known type overridden: got {result['entity_type']!r}"
    )
