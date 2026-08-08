"""
ingestion/extractor.py

Enriches rbi_enforcement_raw.json using Claude Haiku:
  • Resolves entity_type for UNKNOWN / None records
  • Extracts violations  (specific violation clauses)
  • Extracts regulation_refs (statute / section citations)

Input:  data/raw/rbi_enforcement_raw.json
Output: data/extracted/rbi_enforcement_extracted.json

Resume-safe: already-processed prids are skipped on re-run.

Run from rbi_intel/:
    python ingestion/extractor.py
    python ingestion/extractor.py --workers 3 --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from anthropic import Anthropic, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config
from ingestion.validator import EnforcementExtractionOutput

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).parent.parent
RAW_PATH = _ROOT / "data" / "raw" / "rbi_enforcement_raw.json"
OUT_PATH = _ROOT / "data" / "extracted" / "rbi_enforcement_extracted.json"
ERR_LOG  = _ROOT / "logs" / "errors.log"

# ── Entity type whitelist ──────────────────────────────────────────────────────
_VALID_TYPES: frozenset[str] = frozenset({
    "Bank-Private", "Bank-Public", "Bank-UCB", "Bank-Cooperative",
    "Bank-DCCB", "Bank-SCB", "Bank-RRB", "Bank-SFB", "Bank-Payment",
    "NBFC-P2P", "NBFC-ICC", "NBFC-HFC", "NBFC-MFI",
    "ARC", "CIC", "PA", "PPI", "UNKNOWN",
})
# Case-insensitive lookup — guards against Haiku returning "bank-private" vs "Bank-Private"
_TYPE_LOWER: dict[str, str] = {t.lower(): t for t in _VALID_TYPES}

# ── Extraction prompt ──────────────────────────────────────────────────────────
from prompts.extraction import PROMPT_STATIC as _PROMPT_STATIC, build_prompt as _build_prompt


# ── Haiku call (tenacity retry on rate-limit / transient errors) ───────────────
# Only retry on transient errors. BadRequestError (400) is permanent — retrying
# it wastes 12 seconds of backoff and produces the same failure every time.
_TRANSIENT_ERRORS = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)

@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=4),
    reraise=True,
)
def _call_haiku(
    prompt: str,
    client: Anthropic,
    cfg: dict[str, Any],
) -> tuple[EnforcementExtractionOutput, int, int]:
    """Returns (validated output, input_tokens, output_tokens)."""
    rsp = client.messages.create(
        model=cfg["models"]["system_model"],
        max_tokens=cfg["llm"]["max_tokens_extraction"],
        temperature=cfg["llm"]["extraction_temperature"],
        messages=[{"role": "user", "content": prompt}],
    )
    raw: str = rsp.content[0].text.strip()
    # Strip markdown fences if Haiku wraps its JSON in ```
    if raw.startswith("```"):
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("```"))

    parsed: dict[str, Any] = json.loads(raw)
    etype: str = str(parsed.get("entity_type", "UNKNOWN")).strip()
    # Case-insensitive normalisation: "bank-private" -> "Bank-Private"
    etype = _TYPE_LOWER.get(etype.lower(), "UNKNOWN")

    return (
        EnforcementExtractionOutput(
            entity_type=etype,
            violations=[str(v) for v in parsed.get("violations", []) if str(v).strip()],
            regulation_refs=[str(r) for r in parsed.get("regulation_refs", []) if str(r).strip()],
        ),
        rsp.usage.input_tokens,
        rsp.usage.output_tokens,
    )


# ── Per-record enrichment ──────────────────────────────────────────────────────
def _enrich(record: dict[str, Any], client: Anthropic, cfg: dict[str, Any]) -> dict[str, Any]:
    """Enrich one record. Never raises — errors are caught, logged, and returned as metadata."""
    body: str = record.get("body_text", "")
    if not body.strip():
        return {**record, "violations": [], "regulation_refs": [], "input_tokens": 0, "output_tokens": 0}

    current_type: str = record.get("entity_type") or "UNKNOWN"
    prompt = _build_prompt(
        entity_name=record.get("entity_name", ""),
        entity_type=current_type,
        body_text=body,
    )

    try:
        out, in_tok, out_tok = _call_haiku(prompt, client, cfg)
    except Exception as exc:  # noqa: BLE001
        _log_error(record["prid"], exc)
        return {
            **record,
            # Coerce None → "UNKNOWN" so downstream consumers always get a valid string
            "entity_type": record.get("entity_type") or "UNKNOWN",
            "violations": [],
            "regulation_refs": [],
            "extraction_error": str(exc),
            "input_tokens": 0,
            "output_tokens": 0,
        }

    return {
        **record,
        # Only override entity_type when the scraper could not determine it
        "entity_type": out.entity_type if current_type == "UNKNOWN" else current_type,
        "violations": out.violations,
        "regulation_refs": out.regulation_refs,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def _log_error(prid: int, exc: Exception) -> None:
    """Append a structured JSON error entry to logs/errors.log."""
    ERR_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps({
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "module":     "extractor",
        "error_type": type(exc).__name__,
        "query":      f"prid={prid}",
        "traceback":  str(exc),
    })
    with ERR_LOG.open("a", encoding="utf-8") as fh:
        fh.write(entry + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(workers: int | None = None, limit: int | None = None, out_path: Path = OUT_PATH) -> None:
    load_dotenv()  # load ANTHROPIC_API_KEY (and others) from .env when run as a script
    cfg = get_config()
    n_workers: int = workers or cfg["ingestion"].get("extraction_workers", 5)

    raw_data: dict[str, Any] = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = raw_data["records"]
    if limit:
        records = records[:limit]

    # Resume: build set of already-processed prids from any existing output
    done_prids: set[int] = set()
    done_records: list[dict[str, Any]] = []
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        done_records = prev.get("records", [])
        done_prids = {r["prid"] for r in done_records}

    todo = [r for r in records if r["prid"] not in done_prids]
    print(f"Total: {len(records)} | Already done: {len(done_prids)} | To extract: {len(todo)}")
    if not todo:
        print("Nothing to do — output is already up to date.")
        return

    client = Anthropic()
    total_in = total_out = errors = 0
    new_records: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="rbi_enforcement_extraction"):
        mlflow.log_params({
            "model":                  cfg["models"]["system_model"],
            "extraction_temperature": cfg["llm"]["extraction_temperature"],
            "max_tokens_extraction":  cfg["llm"]["max_tokens_extraction"],
            "records_to_process":     len(todo),
            "workers":                n_workers,
        })

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_enrich, r, client, cfg): r["prid"] for r in todo}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()  # _enrich never raises
                new_records.append(result)
                total_in  += result.get("input_tokens", 0)
                total_out += result.get("output_tokens", 0)
                if "extraction_error" in result:
                    errors += 1
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)} processed …")

        all_records = sorted(done_records + new_records, key=lambda r: r["prid"])

        # Atomic write — prevents OneDrive from truncating a mid-stream file
        payload: dict[str, Any] = {
            **raw_data,
            "extraction_metadata": {
                "extracted_at":        datetime.now(timezone.utc).isoformat(),
                "model":               cfg["models"]["system_model"],
                "total_records":       len(all_records),
                "total_input_tokens":  total_in,
                "total_output_tokens": total_out,
                "extraction_errors":   errors,
            },
            "records": all_records,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name
        shutil.move(tmp_path, str(out_path))

        mlflow.log_metrics({
            "total_input_tokens":  total_in,
            "total_output_tokens": total_out,
            "records_processed":   len(new_records),
            "extraction_errors":   errors,
        })

    print(f"\nSaved {len(all_records)} records → {out_path}")
    print(f"Tokens — in: {total_in:,}  out: {total_out:,}  errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RBI enforcement enrichment via Claude Haiku")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override extraction_workers from config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process first N records only (smoke test)")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    main(workers=args.workers, limit=args.limit, out_path=args.output)
