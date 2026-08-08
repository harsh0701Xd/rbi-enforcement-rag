"""
ingestion/graph_writer.py

Writes the RBI enforcement knowledge graph to Neo4j (Schema A).

Schema:
    (Institution)-[:RECEIVED]->(PenaltyAction)-[:FOR_VIOLATION]->(Violation)
                                              -[:CITES]->(Regulation)

Source:  data/validated/rbi_enforcement_validated.json
         Only records where is_bulk_cancellation=False are ingested.

Idempotent: MERGE everywhere — safe to re-run without duplicating data.

Global deduplication:
    Violation  — MERGE ON text_key  (normalised lowercase text)
    Regulation — MERGE ON citation_key (normalised lowercase citation)
    Institution — MERGE ON name; entity_type promoted from UNKNOWN if a better
                  classification exists across multiple penalty actions.

Usage:
    python ingestion/graph_writer.py             # full ingest
    python ingestion/graph_writer.py --limit 20  # smoke test
    python ingestion/graph_writer.py --dry-run   # print row counts, no writes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from ingestion._neo4j import build_driver, execute_cypher_safe
from ingestion._rbi_graph_queries import (
    CONSTRAINTS, DENSITY_CHECK,
    MERGE_INSTITUTIONS, MERGE_PENALTY_ACTIONS, MERGE_REGULATIONS, MERGE_VIOLATIONS,
    REL_CITES, REL_FOR_VIOLATION, REL_RECEIVED,
)

_ROOT      = Path(__file__).parent.parent
_VALIDATED = _ROOT / "data" / "validated" / "rbi_enforcement_validated.json"
_LOG_DIR   = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "graph_writer.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Row builders ──────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Normalisation key for global deduplication: lowercase + strip."""
    return text.lower().strip()


def build_rows(records: list[dict[str, Any]]) -> dict[str, list[dict]]:
    """
    Transform validated records into row-lists for each MERGE/REL batch.

    Violation and Regulation rows are globally deduplicated — the same node
    is shared across all institutions that share the violation text or citation.
    Institution rows are deduplicated by name; UNKNOWN entity_type is promoted
    to a known type if one is encountered later (handled in Cypher SET clause).
    """
    seen_institutions:  set[str] = set()
    seen_violations:    set[str] = set()
    seen_regulations:   set[str] = set()

    inst_rows:    list[dict] = []
    penalty_rows: list[dict] = []
    viol_rows:    list[dict] = []
    reg_rows:     list[dict] = []
    recv_rows:    list[dict] = []
    forv_rows:    list[dict] = []
    cite_rows:    list[dict] = []

    for r in records:
        name = r["entity_name"]

        if name not in seen_institutions:
            inst_rows.append({"name": name, "entity_type": r["entity_type"]})
            seen_institutions.add(name)

        penalty_rows.append({
            "prid":         r["prid"],
            "date":         r["date"],
            "title":        r["title"],
            "penalty_lakh": r.get("penalty_lakh"),
            "action_type":  r["action_type"],
            "source_url":   r["source_url"],
            "pr_number":    r["pr_number"],
        })

        recv_rows.append({"institution_name": name, "prid": r["prid"]})

        for v in r.get("violations", []):
            key = _norm(v)
            if key not in seen_violations:
                viol_rows.append({"text_key": key, "text": v})
                seen_violations.add(key)
            forv_rows.append({"prid": r["prid"], "text_key": key})

        for c in r.get("regulation_refs", []):
            key = _norm(c)
            if key not in seen_regulations:
                reg_rows.append({"citation_key": key, "citation": c})
                seen_regulations.add(key)
            cite_rows.append({"prid": r["prid"], "citation_key": key})

    return {
        "institutions":    inst_rows,
        "penalty_actions": penalty_rows,
        "violations":      viol_rows,
        "regulations":     reg_rows,
        "received":        recv_rows,
        "for_violation":   forv_rows,
        "cites":           cite_rows,
    }


# ── Constraints ───────────────────────────────────────────────────────────────

def _write_constraints(driver: Any) -> None:
    for q in CONSTRAINTS:
        try:
            execute_cypher_safe(q, driver)
        except Exception as exc:
            log.warning("Constraint (may already exist): %s", exc)


# ── Density check ─────────────────────────────────────────────────────────────

def _density_check(driver: Any, cfg: dict) -> None:
    min_degree: int = cfg["graph"]["min_avg_out_degree"]
    rows = execute_cypher_safe(DENSITY_CHECK, driver)
    log.info("── Graph Density ──────────────────────────────")
    for row in rows:
        if not row.get("type"):
            continue
        degree = float(row.get("avg_out_degree") or 0.0)
        # PenaltyAction is the bridge node — its avg_out_degree drives multi-hop
        # traversal quality. Institution only has RECEIVED edges so its structural
        # avg_out is always low; flagging it here produces a false alarm.
        flag = (
            "  ⚠ BELOW THRESHOLD — halt Stage 3"
            if row["type"] == "PenaltyAction" and degree < min_degree
            else ""
        )
        log.info("  %-18s  n=%-5d  avg_out=%.2f%s",
                 row["type"], row["count"], degree, flag)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(limit: int | None = None, dry_run: bool = False) -> None:
    cfg = get_config()

    data = json.loads(_VALIDATED.read_text(encoding="utf-8"))
    records = [r for r in data["records"] if not r["is_bulk_cancellation"]]
    if limit:
        records = records[:limit]

    log.info("Loaded %d usable records (limit=%s)", len(records), limit)
    rows = build_rows(records)

    for k, v in rows.items():
        log.info("  %-18s : %d rows", k, len(v))

    if dry_run:
        log.info("DRY RUN — no writes performed")
        return

    driver = build_driver(cfg)
    try:
        _write_constraints(driver)

        # Nodes first — all must exist before any relationship is created
        for query, key, label in [
            (MERGE_INSTITUTIONS,    "institutions",    "Institution"),
            (MERGE_PENALTY_ACTIONS, "penalty_actions", "PenaltyAction"),
            (MERGE_VIOLATIONS,      "violations",      "Violation"),
            (MERGE_REGULATIONS,     "regulations",     "Regulation"),
        ]:
            execute_cypher_safe(query, driver, {"rows": rows[key]})
            log.info("Wrote %d %s nodes", len(rows[key]), label)

        # Relationships — endpoints guaranteed to exist from node merges above
        for query, key, label in [
            (REL_RECEIVED,      "received",      "RECEIVED"),
            (REL_FOR_VIOLATION, "for_violation", "FOR_VIOLATION"),
            (REL_CITES,         "cites",         "CITES"),
        ]:
            execute_cypher_safe(query, driver, {"rows": rows[key]})
            log.info("Wrote %d %s rels", len(rows[key]), label)

        _density_check(driver, cfg)
    finally:
        driver.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build RBI enforcement Neo4j graph (Schema A).")
    p.add_argument("--limit",   type=int, default=None,
                   help="Process first N records only (smoke test)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print row counts without writing to Neo4j")
    args = p.parse_args()
    main(limit=args.limit, dry_run=args.dry_run)
