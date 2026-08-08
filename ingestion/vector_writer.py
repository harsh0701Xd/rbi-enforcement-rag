"""
ingestion/vector_writer.py

Embeds the RBI enforcement corpus into ChromaDB via Ollama nomic-embed-text.

Reads:  data/validated/rbi_enforcement_validated.json
Writes: ChromaDB collection at $CHROMA_PERSIST_DIR (default: ./data/chroma)
        Collection name from cfg["chroma"]["collection_name"]

Strategy: delete-recreate on every run — guarantees clean state for RAGAS
ablation experiments where chunk_size / chunk_overlap / chars_per_token change
between runs. Stale chunks from a previous config cannot contaminate scores.

Post-run integrity gate: asserts collection.count() == expected total.
If Ollama crashes mid-run, the next full run will rebuild cleanly.

Usage (from rbi_intel/ directory):
    python -m ingestion.vector_writer              # full run
    python -m ingestion.vector_writer --dry-run    # count chunks, no writes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import chromadb
import httpx
import mlflow
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from ingestion._text_processor import OllamaUnavailableError, chunk_text, normalize_entity_name

# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_DIR  = Path(__file__).parent.parent / "logs"
_DATA_DIR = Path(__file__).parent.parent / "data"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "vector_writer.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Embedding via Ollama ──────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=4),
    reraise=True,
)
def _embed_batch(texts: list[str], cfg: dict) -> list[list[float]]:
    """
    POST a batch to Ollama /api/embed (requires Ollama ≥ 0.1.31).
    Raises OllamaUnavailableError on ConnectError — never swallowed silently.
    Retries ×3 on TransportError with exponential backoff.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model    = cfg["embedding"]["model"]
    try:
        resp = httpx.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=cfg["embedding"]["ollama"]["request_timeout"],
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]
    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            f"Ollama unreachable at {base_url} — run `ollama serve` first."
        ) from exc


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _reset_collection(cfg: dict) -> chromadb.Collection:
    """
    Delete the existing collection (if any), then recreate it.
    Guarantees that changing chunk_size/chunk_overlap in config.yaml and
    re-running never leaves old chunks alongside new ones.
    """
    load_dotenv()
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
    col_name    = cfg["chroma"]["collection_name"]
    client      = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection(col_name)
        log.info("Deleted existing collection '%s'", col_name)
    except Exception:
        log.info("No existing collection '%s' — creating fresh", col_name)

    return client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},   # cosine matches nomic-embed-text space
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def write_vectors(cfg: dict, dry_run: bool = False) -> dict[str, int | float]:
    """
    Full RBI enforcement vectorisation pipeline.

    Steps:
      1. Load validated JSON, filter is_bulk_cancellation=False
      2. Chunk body_text via chunk_text() — all params from config.yaml
      3. Delete-recreate ChromaDB collection
      4. Embed in batches via Ollama, add to ChromaDB
      5. Assert collection count == expected (integrity gate)
      6. Log params + metrics to MLflow

    Returns summary dict with records, chunks, batches, elapsed.
    """
    load_dotenv()
    ing  = cfg["ingestion"]
    cpt  = float(ing["chars_per_token"])
    chunk_chars   = int(ing["chunk_size"]      * cpt)
    overlap_chars = int(ing["chunk_overlap"]    * cpt)
    min_chars     = int(ing["min_chunk_length"] * cpt)
    batch_sz      = int(ing["embed_batch_size"])

    # ── 1. Load records ───────────────────────────────────────────────────────
    path    = _DATA_DIR / "validated" / "rbi_enforcement_validated.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [r for r in payload["records"] if not r.get("is_bulk_cancellation", False)]
    log.info("Loaded %d usable records (bulk cancellations excluded)", len(records))

    # ── 2. Chunk all records into flat lists ──────────────────────────────────
    all_ids:   list[str]  = []
    all_texts: list[str]  = []
    all_meta:  list[dict] = []

    for rec in records:
        chunks = chunk_text(
            rec["body_text"],
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
        )
        if not chunks:
            log.warning("prid=%s produced no chunks — skipping", rec["prid"])
            continue

        entity_name = normalize_entity_name(rec["entity_name"])
        total       = len(chunks)

        for i, text in enumerate(chunks):
            all_ids.append(f"{rec['prid']}_chunk_{i}")
            all_texts.append(text)
            all_meta.append({
                "prid":         rec["prid"],
                "entity_name":  entity_name,
                "entity_type":  rec["entity_type"],
                "date":         rec["date"],
                "action_type":  rec["action_type"],
                "source_url":   rec["source_url"],
                "pr_number":    str(rec["pr_number"]),
                "chunk_index":  i,
                "total_chunks": total,
            })

    total_chunks = len(all_ids)
    log.info("Chunking complete: %d records → %d chunks", len(records), total_chunks)

    if dry_run:
        log.info("DRY RUN — no embeddings written to ChromaDB")
        return {"records": len(records), "chunks": total_chunks}

    # ── 3. Delete-recreate collection ─────────────────────────────────────────
    collection = _reset_collection(cfg)

    # ── 4. Embed and write in batches ─────────────────────────────────────────
    t0      = time.monotonic()
    batches = 0
    n_batches_total = -(-total_chunks // batch_sz)   # ceiling division

    for start in range(0, total_chunks, batch_sz):
        end     = min(start + batch_sz, total_chunks)
        vectors = _embed_batch(all_texts[start:end], cfg)
        collection.add(
            ids        = all_ids[start:end],
            embeddings = vectors,
            documents  = all_texts[start:end],
            metadatas  = all_meta[start:end],
        )
        batches += 1
        if batches % 10 == 0:
            log.info("  batch %d / %d", batches, n_batches_total)

    elapsed = round(time.monotonic() - t0, 1)

    # ── 5. Post-run integrity gate ────────────────────────────────────────────
    actual = collection.count()
    if actual < total_chunks:
        raise RuntimeError(
            f"Collection count mismatch after write: expected {total_chunks}, "
            f"got {actual}. Re-run vector_writer to rebuild from scratch."
        )
    log.info("Integrity gate passed: %d chunks in collection ✓", actual)

    # ── 6. MLflow logging ─────────────────────────────────────────────────────
    try:
        mlflow.set_experiment("rbi-intel-vector-ingestion")
        with mlflow.start_run():
            mlflow.log_params({
                "chunk_size":       ing["chunk_size"],
                "chunk_overlap":    ing["chunk_overlap"],
                "chars_per_token":  cpt,
                "min_chunk_length": ing["min_chunk_length"],
                "embed_batch_size": batch_sz,
                "embedding_model":  cfg["embedding"]["model"],
                "collection_name":  cfg["chroma"]["collection_name"],
            })
            mlflow.log_metrics({
                "total_records":   float(len(records)),
                "total_chunks":    float(total_chunks),
                "embed_batches":   float(batches),
                "elapsed_seconds": elapsed,
            })
    except Exception as exc:
        log.warning("MLflow logging failed (non-fatal): %s", exc)

    log.info(
        "Done — %d records · %d chunks · %d batches · %.1fs",
        len(records), total_chunks, batches, elapsed,
    )
    return {
        "records": len(records),
        "chunks":  total_chunks,
        "batches": batches,
        "elapsed": elapsed,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed RBI enforcement corpus into ChromaDB.")
    p.add_argument("--dry-run", action="store_true",
                   help="Count chunks without calling Ollama or writing to ChromaDB.")
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    _cfg    = get_config()
    summary = write_vectors(_cfg, dry_run=args.dry_run)
    print("\n── Vector Write Summary ──")
    for k, v in summary.items():
        print(f"  {k:<16}: {v}")
