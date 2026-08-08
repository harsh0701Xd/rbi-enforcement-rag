"""
rbi_enforcement_scraper.py
--------------------------
Scrapes RBI press releases for monetary-penalty / CoR-cancellation /
cease-and-desist actions (FY20-21 onward) via sequential prid enumeration.

Strategy
--------
* Individual PR pages at BS_PressReleaseDisplay.aspx?prid=N are plain HTML
  (no JavaScript required) — fetched concurrently via ThreadPoolExecutor.
* A binary search locates the prid boundaries for the configured date window
  before the main sweep begins.
* ENFORCEMENT_SIGNALS filters out the ~99% of PRs that are not enforcement.
* Output: data/raw/rbi_enforcement_raw.json

Usage (from rbi_intel/ directory in WSL2)
-----------------------------------------
    pip install requests beautifulsoup4 --break-system-packages
    python ingestion/rbi_enforcement_scraper.py

Or with explicit date range override:
    python ingestion/rbi_enforcement_scraper.py \
        --start-date 2022-04-01 --end-date 2024-03-31

Anchor calibration points (never remove — used for binary search validation)
-----------------------------------------------------------------------------
    prid=49352  →  Feb 06, 2020  (PR 2019-2020/1900)
    prid=52060  →  Aug 13, 2021  (PR 2021-2022/695)
    prid=53539  →  Apr 08, 2022  (PR 2022-2023/41)
    prid=57706  →  Apr 16, 2024  (PR 2024-2025/109)
    prid=57820  →  May 02, 2024  (PR 2024-2025/223)
    prid=58831  →  Oct 07, 2024  (PR 2024-2025/1233)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL: str = (
    "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
)

# Anchor: prid=58831 confirmed Oct 07, 2024 by manual probe.
# Used as the reference point for binary search interpolation.
ANCHOR_PRID: int = 58831
ANCHOR_DATE: date = date(2024, 10, 7)

# Default scrape window.  FY20-21 = Apr 1 2020; adjust via CLI flags.
SCOPE_START: date = date(2020, 4, 1)
SCOPE_END: date = date.today()

# Title substrings that identify an enforcement PR (case-insensitive).
ENFORCEMENT_SIGNALS: tuple[str, ...] = (
    "monetary penalty",
    "imposes penalty",
    "cancels certificate of registration",
    "cancellation of certificate of registration",
    "cease and desist",
    "directions to cease",
    "cancel the certificate",
    "imposed a monetary",
    "penalty imposed",
)

# Concurrent fetch workers.  Keep low to avoid rate-limiting RBI servers.
WORKERS: int = 5

# Per-request settings
REQUEST_TIMEOUT_SECONDS: int = 20
RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF_SECONDS: float = 2.0

# Rough lower and upper prid bounds for binary search (well outside any
# real FY20-21 date — safe to widen if the site grows).
BINARY_SEARCH_LO: int = 40_000
BINARY_SEARCH_HI: int = 75_000

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Regex patterns for extraction
_AMOUNT_RE: re.Pattern[str] = re.compile(
    r"(?:₹|Rs\.?)\s*([\d,]+(?:\.\d+)?)\s*(lakh|crore|lakhs|crores)",
    re.IGNORECASE,
)
_ENTITY_TYPE_RE: re.Pattern[str] = re.compile(
    # Order matters: more-specific patterns before generic ones.
    # Bare abbreviations (CIC, PA, PPI) removed — they cause false-positives
    # in body text (e.g. "45-IC", "Uttar Pradesh", "Jaipur").
    # Full-phrase matches only; Haiku extractor resolves residual UNKNOWN types.
    r"\b(NBFC-P2P|NBFC-ICC|NBFC-HFC|NBFC-MFI|NBFC-ND|NBFC-D"
    r"|Bank-Private|Bank-Public|Bank-Small Finance"
    r"|Bank-Payment|Bank-Co-operative"
    r"|Urban Co-operative Bank|State Co-operative Bank"
    r"|Primary Co-operative Bank|Co-operative Bank"
    r"|Credit Information Company"
    r"|Payment Aggregator|Payment System Operator"
    r"|Prepaid Payment Instrument"
    r"|Asset Reconstruction Company"
    r"|Housing Finance Company"
    r"|Microfinance Institution"
    r"|Gold Loan Company|NBFC)\b",
    re.IGNORECASE,
)
_DATE_RE: re.Pattern[str] = re.compile(
    r"Date\s*:\s*([A-Za-z]+ \d{1,2},\s*\d{4})"
)
_PR_NUM_RE: re.Pattern[str] = re.compile(
    r"Press\s+Release\s*:\s*(\d{4}-\d{4}/\d+)"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EnforcementRecord:
    prid: int
    date: str                     # ISO-8601 YYYY-MM-DD
    title: str
    entity_name: str
    entity_type: Optional[str]
    penalty_lakh: Optional[float]  # None for CoR cancellations / cease-only
    other_action: Optional[str]   # "cease_and_desist" | "cor_cancellation" | None
    violations_raw: list[str]     # raw violation sentences from body
    body_text: str                # full body for downstream extraction
    source_url: str
    pr_number: Optional[str]      # e.g. "2024-2025/1423"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _fetch(prid: int) -> Optional[str]:
    """
    Fetch a single RBI press release page.
    Returns HTML string on success, None on permanent failure.
    """
    url = f"{BASE_URL}?prid={prid}"
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response.text
            if response.status_code in (404, 410):
                # Prid does not exist — not an error, just a gap.
                return None
            _LOG.warning(
                "prid=%d attempt=%d status=%d",
                prid,
                attempt,
                response.status_code,
            )
        except requests.RequestException as exc:
            _LOG.warning("prid=%d attempt=%d error=%s", prid, attempt, exc)
        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_date_and_title(
    soup: BeautifulSoup,
) -> tuple[Optional[date], str]:
    """
    Find the PR date and title from the page.

    RBI pages use <b> tags (not <strong>) for date and title.
    Both list-layout (li > b) and table-layout (td.tableheader > b) pages
    use the same pattern: a <b> whose text begins with "Date :".
    The immediately following <b> in document order is the title.
    We also check <strong> as a fallback in case the layout varies.
    """
    all_strong: list = soup.find_all(["b", "strong"])

    pr_date: Optional[date] = None
    title: str = ""

    for idx, strong_tag in enumerate(all_strong):
        text: str = strong_tag.get_text(strip=True)
        if "Date :" in text or text.startswith("Date :"):
            date_match = _DATE_RE.search(text)
            if date_match:
                try:
                    pr_date = datetime.strptime(
                        date_match.group(1).strip(), "%b %d, %Y"
                    ).date()
                except ValueError:
                    pass
            # Title is the next meaningful <b>/<strong> tag in document order.
            # Skip empty or very short tags (e.g. footnote markers like "**").
            for next_tag in all_strong[idx + 1:]:
                candidate: str = next_tag.get_text(strip=True)
                if len(candidate) > 10 and "Date :" not in candidate:
                    title = candidate
                    break
            break

    return pr_date, title


def _extract_body_and_pr_number(
    soup: BeautifulSoup, title: str
) -> tuple[str, Optional[str]]:
    """
    Extract body text and PR number from the page.

    Strategy: take full page text, locate title, collect everything until
    the PR number line, then return both body and the PR number string.
    """
    full_text: str = soup.get_text(separator="\n", strip=True)

    # Find where the body starts (just after the title).
    title_anchor: str = title[:60] if len(title) > 60 else title
    pos: int = full_text.lower().find(title_anchor.lower())
    if pos == -1:
        return "", None

    after_title: str = full_text[pos + len(title_anchor):]

    # Locate PR number.
    pr_match = _PR_NUM_RE.search(after_title)
    pr_number: Optional[str] = pr_match.group(1) if pr_match else None

    body: str = (
        after_title[: pr_match.start()].strip()
        if pr_match
        else after_title[:3000].strip()
    )

    return body, pr_number


def _is_enforcement(title: str) -> bool:
    """Return True if the PR title matches any enforcement signal."""
    title_lower: str = title.lower()
    return any(signal in title_lower for signal in ENFORCEMENT_SIGNALS)


def _extract_entity_name(title: str) -> str:
    """
    Extract entity name from the PR title.

    RBI titles follow predictable patterns, e.g.:
      "RBI imposes monetary penalty on XYZ Finance Limited"
      "Reserve Bank of India cancels the Certificate of Registration of ABC Ltd"
      "RBI imposes monetary penalty on M/s XYZ Finance Private Limited"

    Returns best-effort entity name; downstream Haiku extraction will refine.
    """
    patterns: list[str] = [
        r"(?:monetary penalty on|penalty on)\s+(.+?)(?:\s+for\b|\s*$)",
        r"(?:cancels?|cancellation of)\s+the\s+Certificate\s+of\s+Registration\s+of\s+(.+?)(?:\s+for\b|\s*\.?\s*$)",
        r"cease\s+and\s+desist\s+(?:directions\s+to\s+)?(.+?)(?:\s+from\b|\s*$)",
        r"directions\s+to\s+cease\s+and\s+desist\s+to\s+(.+?)(?:\s+from\b|\s*$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            name: str = m.group(1).strip().rstrip(".,;")
            # Strip leading M/s, M/S, etc.
            name = re.sub(r"^M/[sS]\s+", "", name).strip()
            return name
    # Fallback: everything after the last preposition sequence
    return title


def _infer_entity_type_from_name(entity_name: str) -> Optional[str]:
    """
    Infer entity type from the entity name when the body text lacks
    an explicit type label.  Covers Hindi/colloquial co-operative bank
    names (Sahakari, Kendriya, etc.) that don't appear in _ENTITY_TYPE_RE.

    Ordered from most-specific to least-specific so longer patterns
    are matched first.
    """
    n: str = entity_name.lower()

    # Urban Co-operative Banks
    if any(x in n for x in ["urban co-operative bank", "urban co operative bank",
                              "urban cooperative bank", "nagri sahakari bank",
                              "nagar sahakari bank", "nagar sahkari bank"]):
        return "Bank-UCB"

    # District Central Co-operative Banks (Kendriya = central in Hindi)
    if any(x in n for x in ["district central co-operative", "kendriya bank",
                              "kendriya sahakari", "zilla sahakari kendriya",
                              "jilla sahakari kendriya", "jila sahakari kendriya"]):
        return "Bank-DCCB"

    # Regional Rural Banks
    if "rural bank" in n or "gramin bank" in n or "kshetriya gramin" in n:
        return "Bank-RRB"

    # Small Finance Banks
    if "small finance bank" in n:
        return "Bank-SFB"

    # Payments Banks
    if "payments bank" in n or "payment bank" in n:
        return "Bank-Payment"

    # Generic Sahakari (co-operative) banks — catches "Sahakari Bank", "Sahkari Bank"
    if any(x in n for x in ["sahakari bank", "sahkari bank", "sahakar bank"]):
        return "Bank-Cooperative"

    # Town banks and generic co-operative banks
    if "town bank" in n or "co-operative bank" in n or "cooperative bank" in n:
        return "Bank-Cooperative"

    # Private/Public sector commercial banks
    if "bank limited" in n or "bank ltd" in n:
        # If name doesn't include co-operative — likely private/public bank
        if "co-operative" not in n and "sahakari" not in n:
            return "Bank-Private"

    return None


def _extract_entity_type(title: str, body: str, entity_name: str = "") -> Optional[str]:
    """
    Match entity category from title + body text, falling back to
    name-pattern inference for co-operative bank variants.
    """
    combined: str = f"{title} {body}"
    m = _ENTITY_TYPE_RE.search(combined)
    if m:
        raw: str = m.group(1)
        # Normalise all variants to canonical EntityCategory names.
        # These must stay consistent — duplicates create orphan nodes in Neo4j.
        normalise: dict[str, str] = {
            # Co-operative bank subtypes
            "co-operative bank": "Bank-Cooperative",
            "primary co-operative bank": "Bank-Cooperative",
            "urban co-operative bank": "Bank-UCB",
            "state co-operative bank": "Bank-SCB",
            # Payment / prepaid
            "payment aggregator": "PA",
            "payment system operator": "PA",
            "prepaid payment instrument": "PPI",
            # NBFC subtypes
            "nbfc": "NBFC-ICC",          # generic catch-all → default ICC subtype
            "gold loan company": "NBFC-ICC",
            "housing finance company": "NBFC-HFC",
            "microfinance institution": "NBFC-MFI",
            "asset reconstruction company": "ARC",
            "credit information company": "CIC",
        }
        return normalise.get(raw.lower(), raw.upper())

    # Regex found nothing — try name-based inference
    if entity_name:
        return _infer_entity_type_from_name(entity_name)
    return None


def _extract_penalty_lakh(body: str) -> Optional[float]:
    """
    Extract penalty amount and normalise to ₹ lakh.
    Returns None for CoR cancellations that carry no monetary penalty.
    """
    m = _AMOUNT_RE.search(body)
    if not m:
        # Also check for "X,XX,XXX" amounts written as plain numbers near "₹"
        plain_m = re.search(
            r"(?:₹|Rs\.?)\s*([\d,]+(?:\.\d+)?)\s*(?=\s)", body, re.IGNORECASE
        )
        if not plain_m:
            return None
        amount_str = plain_m.group(1).replace(",", "")
        amount_float = float(amount_str)
        # If >10,000 it's probably in rupees not lakh; convert
        if amount_float > 10_000:
            return round(amount_float / 100_000, 4)
        return round(amount_float, 4)

    amount_str: str = m.group(1).replace(",", "")
    unit: str = m.group(2).lower()
    amount: float = float(amount_str)

    if "crore" in unit:
        return round(amount * 100, 4)   # crore → lakh
    return round(amount, 4)             # already in lakh


def _extract_other_action(title: str, body: str) -> Optional[str]:
    """Detect cease-and-desist or CoR cancellation as secondary action."""
    combined_lower: str = f"{title} {body}".lower()
    if "cease and desist" in combined_lower or "directions to cease" in combined_lower:
        return "cease_and_desist"
    if (
        "cancels" in combined_lower
        or "cancellation" in combined_lower
        and "certificate of registration" in combined_lower
    ):
        return "cor_cancellation"
    return None


def _extract_violations(body: str) -> list[str]:
    """
    Extract individual violation clauses from the body.

    RBI enforcement PRs typically contain a numbered list of violations
    such as:
        "(i) non-adherence to KYC norms..."
        "(ii) failure to..."
    or plain numbered items: "1. ...", "(a) ...", etc.
    """
    # Match items starting with (i), (ii), (a), (b), 1., 2., etc.
    pattern: re.Pattern[str] = re.compile(
        r"(?:^|\n)\s*(?:\(\s*[ivxlcdmIVXLCDM]+\s*\)"
        r"|\(\s*[a-zA-Z]\s*\)"
        r"|\d+\s*\.)\s*(.+?)(?=(?:\n\s*(?:\(\s*[ivxlcdmIVXLCDM]+\s*\)"
        r"|\(\s*[a-zA-Z]\s*\)|\d+\s*\.))|\Z)",
        re.DOTALL,
    )
    items: list[str] = []
    for m in pattern.finditer(body):
        clause: str = " ".join(m.group(1).split())  # normalise whitespace
        if 10 < len(clause) < 600:  # discard noise
            items.append(clause)
    return items


# ---------------------------------------------------------------------------
# Full record parser
# ---------------------------------------------------------------------------

def parse_enforcement_pr(
    html: str,
    prid: int,
) -> Optional[EnforcementRecord]:
    """
    Parse a confirmed enforcement PR HTML page into a structured record.
    Returns None if required fields cannot be extracted.
    """
    soup = BeautifulSoup(html, "html.parser")
    pr_date, title = _extract_date_and_title(soup)

    if not pr_date or not title:
        _LOG.warning("prid=%d: could not extract date or title", prid)
        return None

    if not _is_enforcement(title):
        return None  # Binary-search artefact — skip quietly

    body, pr_number = _extract_body_and_pr_number(soup, title)

    entity_name: str = _extract_entity_name(title)

    record = EnforcementRecord(
        prid=prid,
        date=pr_date.isoformat(),
        title=title,
        entity_name=entity_name,
        entity_type=_extract_entity_type(title, body, entity_name),
        penalty_lakh=_extract_penalty_lakh(body),
        other_action=_extract_other_action(title, body),
        violations_raw=_extract_violations(body),
        body_text=body[:4000],  # cap at 4 KB per record
        source_url=f"{BASE_URL}?prid={prid}",
        pr_number=pr_number,
    )
    return record


# ---------------------------------------------------------------------------
# Binary search for prid boundaries
# ---------------------------------------------------------------------------

def _prid_to_date(prid: int) -> Optional[date]:
    """Fetch prid and return its publication date, or None on failure."""
    html = _fetch(prid)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    pr_date, _ = _extract_date_and_title(soup)
    return pr_date


def _binary_search_prid(
    target: date,
    lo: int = BINARY_SEARCH_LO,
    hi: int = BINARY_SEARCH_HI,
    find_first: bool = True,
) -> int:
    """
    Binary search to find the prid whose publication date is closest to
    `target`.

    find_first=True  → returns the lowest prid with date >= target (start bound)
    find_first=False → returns the highest prid with date <= target (end bound)

    The search fetches O(log N) pages; at N=35,000 that is at most 16 fetches.
    """
    _LOG.info(
        "Binary search for prid corresponding to %s (lo=%d hi=%d find_first=%s)",
        target,
        lo,
        hi,
        find_first,
    )
    best: int = hi if find_first else lo

    while lo <= hi:
        mid: int = (lo + hi) // 2
        mid_date: Optional[date] = _prid_to_date(mid)

        if mid_date is None:
            # Page missing — step past it
            lo = mid + 1
            continue

        _LOG.debug("  prid=%d → %s", mid, mid_date)

        if find_first:
            if mid_date >= target:
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if mid_date <= target:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

    _LOG.info("Binary search result: prid=%d", best)
    return best


def find_scope_prids(
    start: date = SCOPE_START,
    end: date = SCOPE_END,
) -> tuple[int, int]:
    """
    Return (start_prid, end_prid) for the configured date window.
    Uses the anchor calibration to seed the binary search bounds so
    fewer HTTP requests are needed.
    """
    # Estimate prids from anchor using a conservative 5.5 prids/day rate.
    PRIDS_PER_DAY: float = 5.5

    days_from_anchor_to_start: int = (ANCHOR_DATE - start).days
    lo_start: int = max(
        BINARY_SEARCH_LO,
        ANCHOR_PRID - int(days_from_anchor_to_start * PRIDS_PER_DAY * 1.5),
    )
    hi_start: int = ANCHOR_PRID - int(
        days_from_anchor_to_start * PRIDS_PER_DAY * 0.5
    )

    days_from_anchor_to_end: int = (end - ANCHOR_DATE).days
    lo_end: int = ANCHOR_PRID + int(days_from_anchor_to_end * PRIDS_PER_DAY * 0.5)
    hi_end: int = min(
        BINARY_SEARCH_HI,
        ANCHOR_PRID + int(days_from_anchor_to_end * PRIDS_PER_DAY * 1.5),
    )

    _LOG.info("Locating start prid (target %s, lo=%d, hi=%d)…", start, lo_start, hi_start)
    start_prid: int = _binary_search_prid(start, lo=lo_start, hi=hi_start, find_first=True)

    _LOG.info("Locating end prid (target %s, lo=%d, hi=%d)…", end, lo_end, hi_end)
    end_prid: int = _binary_search_prid(end, lo=lo_end, hi=hi_end, find_first=False)

    _LOG.info("Scope: prid %d to %d (%d prids to scan)", start_prid, end_prid, end_prid - start_prid)
    return start_prid, end_prid


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def _process_prid(prid: int) -> Optional[EnforcementRecord]:
    """
    Worker function: fetch one prid, filter non-enforcement, parse if matched.
    Called concurrently from the thread pool.
    """
    html = _fetch(prid)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    _, title = _extract_date_and_title(soup)

    if not title or not _is_enforcement(title):
        return None  # Fast path: skip 99% of PRs without full parse

    _LOG.info("MATCH  prid=%d  title=%s", prid, title[:80])
    return parse_enforcement_pr(html, prid)


def scrape(
    start_prid: int,
    end_prid: int,
    workers: int = WORKERS,
) -> list[EnforcementRecord]:
    """
    Enumerate all prids in [start_prid, end_prid] concurrently and
    return matched enforcement records sorted by date ascending.
    """
    total: int = end_prid - start_prid + 1
    _LOG.info("Sweeping %d prids with %d workers…", total, workers)

    records: list[EnforcementRecord] = []
    processed: int = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_prid, prid): prid
            for prid in range(start_prid, end_prid + 1)
        }
        for future in as_completed(futures):
            processed += 1
            if processed % 200 == 0:
                _LOG.info(
                    "Progress: %d/%d (%.1f%%) — %d enforcement found",
                    processed,
                    total,
                    100 * processed / total,
                    len(records),
                )
            try:
                result = future.result()
                if result is not None:
                    records.append(result)
            except Exception as exc:
                prid = futures[future]
                _LOG.error("prid=%d unexpected error: %s", prid, exc)

    records.sort(key=lambda r: r.date)
    _LOG.info("Sweep complete. %d enforcement PRs found.", len(records))
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape RBI enforcement press releases"
    )
    parser.add_argument(
        "--start-date",
        default=SCOPE_START.isoformat(),
        help="Earliest date to include (YYYY-MM-DD, default: %(default)s)",
    )
    parser.add_argument(
        "--end-date",
        default=SCOPE_END.isoformat(),
        help="Latest date to include (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--start-prid",
        type=int,
        default=None,
        help="Override start prid (skips binary search)",
    )
    parser.add_argument(
        "--end-prid",
        type=int,
        default=None,
        help="Override end prid (skips binary search)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="Concurrent fetch workers (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/rbi_enforcement_raw.json",
        help="Output JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--smoke-test",
        type=int,
        default=0,
        metavar="N",
        help="If >0, scan only N prids centred around the anchor (dev testing)",
    )
    args = parser.parse_args()

    start_date: date = date.fromisoformat(args.start_date)
    end_date: date = date.fromisoformat(args.end_date)

    if args.smoke_test > 0:
        half: int = args.smoke_test // 2
        start_prid: int = ANCHOR_PRID - half
        end_prid: int = ANCHOR_PRID + half
        _LOG.info(
            "Smoke-test mode: scanning prids %d–%d (%d total)",
            start_prid,
            end_prid,
            args.smoke_test,
        )
    elif args.start_prid and args.end_prid:
        start_prid = args.start_prid
        end_prid = args.end_prid
    else:
        start_prid, end_prid = find_scope_prids(start_date, end_date)

    records: list[EnforcementRecord] = scrape(
        start_prid, end_prid, workers=args.workers
    )

    output_path: Path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "metadata": {
            "scraped_at": datetime.utcnow().isoformat() + "Z",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "start_prid": start_prid,
            "end_prid": end_prid,
            "total_enforcement_records": len(records),
        },
        "records": [asdict(r) for r in records],
    }

    # Atomic write: serialise to /tmp first, then move into the OneDrive
    # folder in one rename() call.  Avoids OneDrive truncating the file
    # while json.dump is still streaming bytes.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp_path: str = tmp.name

    shutil.move(tmp_path, output_path)
    _LOG.info("Saved %d records to %s", len(records), output_path)


if __name__ == "__main__":
    main()
