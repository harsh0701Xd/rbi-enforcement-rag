"""
rag/_logger.py

Per-query inference logging.  Two destinations:
  1. logs/inference.jsonl  — one JSON record per question, always written
  2. MLflow experiment rbi-intel-inference — optional, fires only when
     MLFLOW_TRACKING_URI env var is set; silently skipped otherwise

Usage (inside any stage's answer() function):
    from rag._logger import InferenceLogger

    logger = InferenceLogger("naive", question)
    ...
    logger.step("embed_retrieve", latency_s=0.31, n_chunks=5, top_score=0.82)
    logger.step("synthesis",      latency_s=1.44, input_tokens=820, output_tokens=310)
    logger.finish(answer_text, n_contexts=5)

Each step() call is stamped with elapsed time since the logger was created.
finish() closes the record, writes JSONL, and triggers MLflow.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_LOG_FILE = Path(__file__).parent.parent / "logs" / "inference.jsonl"


class InferenceLogger:
    """Accumulates per-step metrics for one pipeline call."""

    def __init__(self, stage: str, question: str) -> None:
        self.stage    = stage
        self.question = question
        self._t0      = time.perf_counter()
        self._steps: list[dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, name: str, **kwargs: Any) -> None:
        """
        Record one named pipeline step.

        kwargs can be any numeric, bool, or short-string values:
          latency_s, n_chunks, top_score, input_tokens, filter_applied, ...

        float values are rounded to 4 decimal places for compact JSON.
        """
        entry: dict[str, Any] = {
            "step":      name,
            "elapsed_s": round(time.perf_counter() - self._t0, 3),
        }
        for k, v in kwargs.items():
            entry[k] = round(v, 4) if isinstance(v, float) else v
        self._steps.append(entry)

    def finish(self, answer: str, n_contexts: int, **extra: Any) -> dict[str, Any]:
        """
        Close the record, write to JSONL, try MLflow.

        Args:
            answer:     The final answer string (length is logged, not content).
            n_contexts: Total context passages seen by the synthesiser.
            **extra:    Any additional top-level fields (e.g. n_graph_rows=7).

        Returns:
            The full record dict (useful for tests / debug prints).
        """
        record: dict[str, Any] = {
            "ts":              datetime.now(timezone.utc).isoformat(),
            "stage":           self.stage,
            "question":        self.question[:300],
            "answer_chars":    len(answer),
            "n_contexts":      n_contexts,
            "total_latency_s": round(time.perf_counter() - self._t0, 3),
            "steps":           self._steps,
            **extra,
        }
        _append_jsonl(record)
        _try_mlflow(record)
        return record


# ── JSONL writer ──────────────────────────────────────────────────────────────

def _append_jsonl(record: dict[str, Any]) -> None:
    """Append one JSON line to logs/inference.jsonl.  Never raises."""
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("inference.jsonl write failed: %s", exc)


# ── MLflow (fire-and-forget) ──────────────────────────────────────────────────

def _try_mlflow(record: dict[str, Any]) -> None:
    """
    Log key metrics to MLflow experiment rbi-intel-inference.

    Requires MLFLOW_TRACKING_URI env var.  All errors are swallowed so
    MLflow unavailability can never crash a live query.

    Metric naming convention: {step_name}__{metric_key}
      e.g. embed_retrieve__n_chunks, synthesis__input_tokens
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return
    try:
        # Fail fast: 0 retries, 5 s timeout so a dead server never blocks a query.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        import mlflow  # lazy — not required at module import time
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("rbi-intel-inference")
        with mlflow.start_run(run_name=f"{record['stage']}_{record['ts'][:10]}"):
            params: dict[str, str] = {
                "stage":    record["stage"],
                "question": record["question"][:250],
            }
            # Log execution mode as a param (not a metric — it's categorical)
            if "mode" in record:
                params["mode"] = str(record["mode"])
            mlflow.log_params(params)
            metrics: dict[str, float] = {
                "total_latency_s": record["total_latency_s"],
                "n_contexts":      float(record["n_contexts"]),
                "answer_chars":    float(record["answer_chars"]),
            }
            # Flatten per-step numerics: embed_retrieve__n_chunks, etc.
            for s in record["steps"]:
                prefix = s["step"]
                for k, v in s.items():
                    if k not in ("step", "elapsed_s") and isinstance(v, (int, float)):
                        metrics[f"{prefix}__{k}"] = float(v)
            # Any top-level numeric extras — exclude string fields already logged as params
            _non_metric = {"ts", "stage", "question", "answer_chars",
                           "n_contexts", "total_latency_s", "steps", "mode"}
            for k, v in record.items():
                if k not in _non_metric and isinstance(v, (int, float)):
                    metrics[k] = float(v)
            mlflow.log_metrics(metrics)
    except Exception as exc:
        log.debug("MLflow inference logging skipped: %s", exc)
