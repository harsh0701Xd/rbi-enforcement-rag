"""
scripts/patch_penalty_amounts.py

One-time data quality patch:
  - Fetches the 178 PenaltyAction nodes where penalty_lakh IS NULL
  - Fetches each press release page from source_url
  - Extracts the penalty amount via regex
  - Sets penalty_lakh (float, in lakhs) and corrects action_type to "monetary_penalty"
  - Logs failures to patch_penalty_failures.log for manual review

Usage (from rbi_intel/ directory):
    python scripts/patch_penalty_amounts.py

Requirements: neo4j, requests (both in project venv)
"""

from __future__ import annotations

import os
import re
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

import requests
from neo4j import GraphDatabase

# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USERNAME"]
NEO4J_PASS = os.environ["NEO4J_PASSWORD"]

LOG_FILE = Path(__file__).parent / "patch_penalty_failures.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

FETCH_DELAY   = 0.6   # seconds between requests
FETCH_TIMEOUT = 15    # seconds per request


# ── Amount extraction ──────────────────────────────────────────────────────────

# Matches the context phrase so we don't pick up unrelated numbers
_CONTEXT_RE = re.compile(
    r"monetary penalty of\s*"
    r"(?:₹|Rs\.?\s*)"
    r"([\d,]+(?:\.\d+)?)"
    r"\s*(lakh|crore|/-|)",
    re.IGNORECASE,
)

# Fallback: any ₹/Rs amount near "penalty" in the page
_FALLBACK_RE = re.compile(
    r"(?:₹|Rs\.?\s*)([\d,]+(?:\.\d+)?)\s*(lakh|crore|/-|)",
    re.IGNORECASE,
)


def _parse_amount(raw_number: str, unit: str) -> float | None:
    """
    Convert raw number string + unit to lakh (float).

    Unit mapping:
      "lakh" or ""  with number < 1000  → already in lakh, use directly
      "/-"  or ""   with number ≥ 1000  → treat as absolute rupees → ÷ 100000
      "crore"                            → × 100
    """
    try:
        val = float(raw_number.replace(",", ""))
    except ValueError:
        return None

    unit = unit.strip().lower()

    if unit == "crore":
        return round(val * 100, 4)
    elif unit == "lakh":
        return round(val, 4)
    elif val >= 1000:
        # Raw rupee amount (e.g. 50000 → 0.5 lakh)
        return round(val / 100_000, 4)
    else:
        # Small number with no unit — ambiguous; treat as lakh
        return round(val, 4)


def extract_amount(html: str) -> float | None:
    """Try context-anchored regex first, then broader fallback."""
    for pattern in (_CONTEXT_RE, _FALLBACK_RE):
        m = pattern.search(html)
        if m:
            amount = _parse_amount(m.group(1), m.group(2))
            if amount is not None and amount > 0:
                return amount
    return None


# ── Neo4j helpers ──────────────────────────────────────────────────────────────

def get_null_records(driver) -> list[dict]:
    """Return all PenaltyAction nodes where penalty_lakh IS NULL."""
    query = """
    MATCH (p:PenaltyAction)
    WHERE p.penalty_lakh IS NULL AND p.source_url IS NOT NULL
    RETURN p.prid AS prid, p.source_url AS url
    ORDER BY p.prid
    """
    with driver.session() as s:
        return [dict(r) for r in s.run(query)]


def patch_record(driver, prid: int, penalty_lakh: float) -> None:
    query = """
    MATCH (p:PenaltyAction {prid: $prid})
    SET p.penalty_lakh  = $penalty_lakh,
        p.action_type   = "monetary_penalty"
    """
    with driver.session() as s:
        s.run(query, prid=prid, penalty_lakh=penalty_lakh)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    records = get_null_records(driver)
    total   = len(records)
    log.info("Found %d PenaltyAction nodes with null penalty_lakh", total)

    session  = requests.Session()
    session.headers.update(HEADERS)

    patched   = 0
    failed    = []

    for i, rec in enumerate(records, 1):
        prid = rec["prid"]
        url  = rec["url"]
        log.info("[%d/%d] prid=%s  %s", i, total, prid, url)

        # Fetch press release page
        try:
            resp = session.get(url, timeout=FETCH_TIMEOUT)
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:
            log.warning("  FETCH FAILED: %s", exc)
            failed.append({"prid": prid, "url": url, "reason": f"fetch: {exc}"})
            time.sleep(FETCH_DELAY)
            continue

        # Extract amount
        amount = extract_amount(html)
        if amount is None:
            log.warning("  PARSE FAILED: no amount found in page")
            failed.append({"prid": prid, "url": url, "reason": "no amount matched"})
            time.sleep(FETCH_DELAY)
            continue

        # Patch Neo4j
        try:
            patch_record(driver, prid, amount)
            log.info("  ✓  penalty_lakh = %.4f", amount)
            patched += 1
        except Exception as exc:
            log.error("  NEO4J WRITE FAILED: %s", exc)
            failed.append({"prid": prid, "url": url, "reason": f"neo4j: {exc}"})

        time.sleep(FETCH_DELAY)

    driver.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("═" * 60)
    log.info("PATCH COMPLETE")
    log.info("  Total records : %d", total)
    log.info("  Patched       : %d", patched)
    log.info("  Failed        : %d", len(failed))

    if failed:
        log.info("")
        log.info("FAILURES — review manually:")
        for f in failed:
            log.info("  prid=%-8s  reason=%-30s  %s", f["prid"], f["reason"], f["url"])
        log.info("")
        log.info("Full failure log saved to: %s", LOG_FILE)


if __name__ == "__main__":
    main()
