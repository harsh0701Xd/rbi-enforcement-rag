"""
scripts/patch_institution_states.py

One-time data quality patch:
  - Reads institution_state_mapping.csv (entity_type, name, state)
  - For each row with a non-null state, MATCHes the Institution node by name
    and SETs i.state = <state>
  - Reports totals and any unmatched names

Usage (from rbi_intel/ directory):
    python scripts/patch_institution_states.py

Requirements: neo4j (in project venv)

CSV expected at:  <project_root>/institution_state_mapping.csv
  i.e. one level above rbi_intel/
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USERNAME"]
NEO4J_PASS = os.environ["NEO4J_PASSWORD"]

# CSV is at FinGraph Intelligence System/institution_state_mapping.csv
# Script is at rbi_intel/scripts/ → go up 3 levels to reach project root
CSV_PATH = Path(__file__).parent.parent.parent / "institution_state_mapping.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ── Neo4j helpers ──────────────────────────────────────────────────────────────

def patch_state(driver, name: str, state: str) -> int:
    """
    SET i.state on Institution node matched by exact name.
    Returns number of nodes updated (0 if no match).
    """
    query = """
    MATCH (i:Institution {name: $name})
    SET i.state = $state
    RETURN count(i) AS updated
    """
    with driver.session() as s:
        result = s.run(query, name=name, state=state)
        record = result.single()
        return record["updated"] if record else 0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CSV_PATH.exists():
        log.error("CSV not found at: %s", CSV_PATH)
        return

    # Load CSV
    rows: list[dict] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    log.info("Loaded %d rows from %s", len(rows), CSV_PATH.name)

    # Filter to rows with a usable state value
    patchable = [r for r in rows if r.get("state") and r["state"].strip()]
    skipped   = len(rows) - len(patchable)
    log.info("  Rows with state: %d  |  Skipped (null state): %d", len(patchable), skipped)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    patched    = 0
    unmatched  = []

    for r in patchable:
        name  = r["name"].strip()
        state = r["state"].strip()
        updated = patch_state(driver, name, state)
        if updated > 0:
            patched += 1
        else:
            unmatched.append(name)

    driver.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("═" * 60)
    log.info("PATCH COMPLETE")
    log.info("  Total rows with state : %d", len(patchable))
    log.info("  Institutions patched  : %d", patched)
    log.info("  Unmatched (no node)   : %d", len(unmatched))

    if unmatched:
        log.info("")
        log.info("Unmatched names (no Institution node found):")
        for name in unmatched:
            log.info("  - %s", name)

    # Verify
    log.info("")
    log.info("Run this in Neo4j Browser to verify:")
    log.info("  MATCH (i:Institution) WHERE i.state IS NOT NULL RETURN COUNT(i)")
    log.info("  MATCH (i:Institution) WHERE i.state IS NULL RETURN COUNT(i)")


if __name__ == "__main__":
    main()
