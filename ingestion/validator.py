"""
ingestion/validator.py

Pydantic v2 models for the RBI enforcement corpus.
This is the single source of truth for the data schema — no inline Pydantic
class definitions anywhere else in the codebase per CLAUDE.md.

Models
------
EnforcementRecord           — fully validated RBI enforcement press release
EnforcementExtractionOutput — Haiku extraction output (used by extractor.py)

Run as a script to validate the extracted corpus:
    python ingestion/validator.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).parent.parent
_EXTRACTED    = _ROOT / "data" / "extracted" / "rbi_enforcement_extracted.json"
_VALIDATED    = _ROOT / "data" / "validated"  / "rbi_enforcement_validated.json"
_ERR_LOG      = _ROOT / "logs" / "errors.log"


# ── RBI Enforcement Models ────────────────────────────────────────────────────

class EnforcementExtractionOutput(BaseModel):
    """
    Haiku extraction output for one RBI enforcement press release.
    Populated by extractor.py. Not used elsewhere per CLAUDE.md.
    """
    entity_type: str
    violations: list[str] = Field(default_factory=list)
    regulation_refs: list[str] = Field(default_factory=list)


class EnforcementRecord(BaseModel):
    """
    Fully validated RBI enforcement press release record.
    Written to data/validated/ by this module.
    Consumed by graph_writer.py (Neo4j) and vector_writer.py (ChromaDB).

    Derived fields (not in raw JSON — computed during validation):
        action_type          — "monetary_penalty" | "cor_cancellation" | "other"
        is_bulk_cancellation — True when the press release covers multiple unnamed
                               institutions (CoR cancellation batches). These records
                               are excluded from graph/vector ingestion because
                               entity_name is a press release headline, not an
                               institution name.
    """
    prid: int
    date: str                              # YYYY-MM-DD
    title: str
    entity_name: str
    entity_type: str                       # canonical type from extractor._VALID_TYPES
    penalty_lakh: Optional[float] = None   # INR lakhs; None for non-monetary actions
    action_type: str                       # derived — see above
    source_url: str
    pr_number: str
    body_text: str
    violations: list[str] = Field(default_factory=list)
    regulation_refs: list[str] = Field(default_factory=list)
    is_bulk_cancellation: bool = False
    input_tokens: int = 0
    output_tokens: int = 0



# ── Validation helpers ────────────────────────────────────────────────────────

def _derive_action_type(raw: dict) -> str:
    if raw.get("other_action") == "cor_cancellation":
        return "cor_cancellation"
    if raw.get("penalty_lakh") is not None:
        return "monetary_penalty"
    return "other"


def _derive_is_bulk(raw: dict) -> bool:
    """
    Bulk CoR cancellation press releases use the headline as entity_name
    ("RBI cancels Certificate of Registration of Fourteen NBFCs") rather than
    a named institution. Both signals are checked for robustness.
    """
    return (
        raw.get("other_action") == "cor_cancellation"
        or raw.get("entity_name") == raw.get("title")
    )


def _to_record(raw: dict) -> EnforcementRecord:
    return EnforcementRecord(
        prid=raw["prid"],
        date=raw["date"],
        title=raw["title"],
        entity_name=raw["entity_name"],
        entity_type=raw.get("entity_type") or "UNKNOWN",
        penalty_lakh=raw.get("penalty_lakh"),
        action_type=_derive_action_type(raw),
        source_url=raw["source_url"],
        pr_number=str(raw["pr_number"]),
        body_text=raw["body_text"],
        violations=raw.get("violations", []),
        regulation_refs=raw.get("regulation_refs", []),
        is_bulk_cancellation=_derive_is_bulk(raw),
        input_tokens=raw.get("input_tokens", 0),
        output_tokens=raw.get("output_tokens", 0),
    )


def _log_error(prid: int, exc: Exception) -> None:
    _ERR_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps({
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "module":     "validator",
        "error_type": type(exc).__name__,
        "query":      f"prid={prid}",
        "traceback":  str(exc),
    })
    with _ERR_LOG.open("a", encoding="utf-8") as fh:
        fh.write(entry + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    raw_data = json.loads(_EXTRACTED.read_text(encoding="utf-8"))
    raw_records: list[dict] = raw_data["records"]

    validated: list[dict] = []
    errors = bulk = 0

    for raw in raw_records:
        try:
            record = _to_record(raw)
        except Exception as exc:
            _log_error(raw.get("prid", -1), exc)
            errors += 1
            continue
        if record.is_bulk_cancellation:
            bulk += 1
        validated.append(record.model_dump())

    usable = len(validated) - bulk
    print(f"Total     : {len(raw_records)}")
    print(f"Valid     : {len(validated)}  (errors: {errors})")
    print(f"Bulk CoR  : {bulk}  (excluded from graph/vector)")
    print(f"Usable    : {usable}  (single-institution records)")

    payload = {
        **{k: v for k, v in raw_data.items() if k != "records"},
        "validation_metadata": {
            "validated_at":      datetime.now(timezone.utc).isoformat(),
            "total_records":     len(raw_records),
            "valid_records":     len(validated),
            "bulk_cancellations": bulk,
            "validation_errors": errors,
            "usable_for_graph":  usable,
        },
        "records": validated,
    }
    _VALIDATED.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    shutil.move(tmp_path, str(_VALIDATED))
    print(f"\nSaved → {_VALIDATED}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
