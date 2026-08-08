"""
eval/dry_run.py

Pre-flight check for the RAGAS evaluation pipeline.

Costs ~18 Claude Sonnet judge calls (3 mock answers x 6 metrics + overhead).
Run this BEFORE eval/ragas_eval.py to catch every failure mode cheaply.

Validates:
  Phase 1 -- Package imports and LLM/embedding wiring (all 6 metrics)
  Phase 2 -- RAGAS evaluate() output format and .items() iteration
  Phase 3 -- JSON file write, NaN handling, MLflow graceful skip
  Phase 4 -- InferenceLogger logs/inference.jsonl write

Usage:
    cd rbi_intel/
    ./venv/bin/python eval/dry_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure rbi_intel/ is on sys.path when the script is run as eval/dry_run.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import math
import os
import tempfile
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

PASS = "[PASS]"
FAIL = "[FAIL]"
_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    if ok:
        log.info("%s  %s  %s", PASS, label, detail)
    else:
        log.error("%s  %s  %s", FAIL, label, detail)
        _failures.append(label)
    return ok


# ---------------------------------------------------------------------------
# Phase 1 -- Imports and wiring
# ---------------------------------------------------------------------------
log.info("=" * 60)
log.info("Phase 1: Package imports and LLM/embedding wiring")
log.info("=" * 60)

try:
    from dotenv import load_dotenv
    load_dotenv()
    check("dotenv import + load", True)
except Exception as e:
    check("dotenv import + load", False, str(e)); sys.exit(1)

try:
    from config import get_config
    cfg = get_config()
    judge_model  = cfg["models"]["judge_model"]
    system_model = cfg["models"]["system_model"]
    check("config.get_config()", True, f"judge={judge_model!r}  system={system_model!r}")
except Exception as e:
    check("config.get_config()", False, str(e)); sys.exit(1)

try:
    from datasets import Dataset
    check("datasets import", True)
except Exception as e:
    check("datasets import", False, str(e)); sys.exit(1)

try:
    from langchain_anthropic import ChatAnthropic
    check("langchain_anthropic import", True)
except Exception as e:
    check("langchain_anthropic import", False, str(e)); sys.exit(1)

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    check("langchain_community.embeddings import", True)
except Exception as e:
    check("langchain_community.embeddings import", False, str(e)); sys.exit(1)

try:
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_entity_recall,
        context_precision,
        context_recall,
        faithfulness,
    )
    check("ragas imports (6 orthogonal metrics)", True)
except Exception as e:
    check("ragas imports", False, str(e)); sys.exit(1)

_ALL_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    context_entity_recall,
]

try:
    _t = time.perf_counter()
    llm = LangchainLLMWrapper(ChatAnthropic(model=judge_model, temperature=0))
    check("LangchainLLMWrapper(ChatAnthropic)", True, f"{time.perf_counter()-_t:.2f}s")
except Exception as e:
    check("LangchainLLMWrapper(ChatAnthropic)", False, str(e)); sys.exit(1)

try:
    _t = time.perf_counter()
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    check("LangchainEmbeddingsWrapper(MiniLM)", True,
          f"{time.perf_counter()-_t:.2f}s (may download on first run)")
except Exception as e:
    check("LangchainEmbeddingsWrapper(MiniLM)", False, str(e)); sys.exit(1)

try:
    for m in _ALL_METRICS:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = emb
    check("All 6 metrics wired (LLM + embeddings)", True,
          str([m.name for m in _ALL_METRICS]))
except Exception as e:
    check("Metric LLM/embedding wiring", False, str(e)); sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 2 -- RAGAS evaluate() with 3 mock results
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
log.info("Phase 2: RAGAS evaluate() on 3 mock answers (~18 Sonnet calls)")
log.info("=" * 60)

MOCK_QUESTIONS = [
    "Has RBI imposed any monetary penalty on Axis Bank?",
    "Which banks were penalised by RBI for KYC non-compliance in 2023?",
    "How many NBFCs received RBI enforcement actions in 2022?",
]
MOCK_ANSWERS = [
    "Yes, RBI imposed a monetary penalty of Rs 0.93 crore on Axis Bank Limited for "
    "non-compliance with RBI directions on Know Your Customer (KYC). "
    "(Axis Bank Limited, 2023-11-14, PR: 2023-2024/1103)",

    "Several banks were penalised by RBI for KYC non-compliance in 2023, including "
    "Axis Bank (Rs 0.93 crore) and certain cooperative banks.",

    "Based on the available context, multiple NBFCs received RBI enforcement actions "
    "in 2022. The exact count may be incomplete as some penalty amounts were not recorded.",
]
MOCK_CONTEXTS = [
    ["RBI imposed a monetary penalty of Rs 0.93 crore on Axis Bank Limited on "
     "November 14, 2023 for non-compliance with directions on Know Your Customer (KYC)."],

    ["RBI imposed a penalty of Rs 0.93 crore on Axis Bank for KYC violations in 2023.",
     "Cooperative banks were also fined by RBI for KYC non-compliance during 2023."],

    ["In 2022, RBI imposed penalties on several NBFCs for regulatory violations. "
     "Bajaj Finance Limited was penalised for non-compliance with KYC directions."],
]
MOCK_GROUND_TRUTHS = [
    "Yes, RBI imposed a monetary penalty of Rs 0.93 crore on Axis Bank for KYC non-compliance.",
    "Banks including Axis Bank were penalised for KYC non-compliance in 2023.",
    "Multiple NBFCs received RBI enforcement actions in 2022.",
]

try:
    dataset = Dataset.from_dict({
        "question":     MOCK_QUESTIONS,
        "answer":       MOCK_ANSWERS,
        "contexts":     MOCK_CONTEXTS,
        "ground_truth": MOCK_GROUND_TRUTHS,
    })
    check("Dataset.from_dict()", True, f"rows={len(dataset)}")
except Exception as e:
    check("Dataset.from_dict()", False, str(e)); sys.exit(1)

try:
    _t = time.perf_counter()
    eval_out = evaluate(dataset, metrics=_ALL_METRICS)
    elapsed = time.perf_counter() - _t
    check("ragas.evaluate() completed", True, f"{elapsed:.1f}s")
except Exception as e:
    check("ragas.evaluate() completed", False, str(e)); sys.exit(1)

try:
    scores = {k: float(v) for k, v in eval_out.items()}
    check("eval_out.items() iterable", True, f"keys={list(scores.keys())}")
except Exception as e:
    check("eval_out.items() iterable", False, str(e)); sys.exit(1)

try:
    expected = {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "answer_correctness", "context_entity_recall",
    }
    actual = set(scores.keys())
    ok = expected.issubset(actual)
    check("All 6 metric names present", ok,
          f"expected={sorted(expected)}  actual={sorted(actual)}")
except Exception as e:
    check("All 6 metric names present", False, str(e))

try:
    for name, val in scores.items():
        assert isinstance(val, float), f"{name}={val!r} is not float"
    check("All scores are float", True, str({k: round(v, 4) for k, v in scores.items()}))
except Exception as e:
    check("All scores are float", False, str(e))

nan_count = sum(1 for v in scores.values() if math.isnan(v))
check("No NaN scores", nan_count == 0,
      f"nan_count={nan_count}" if nan_count else "all finite")


# ---------------------------------------------------------------------------
# Phase 3 -- JSON write, NaN guard, MLflow graceful skip
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
log.info("Phase 3: File output and MLflow graceful skip")
log.info("=" * 60)

with tempfile.TemporaryDirectory() as td:
    out_path = Path(td) / "dry_run_scores.json"
    try:
        safe_scores = {k: (0.0 if math.isnan(v) or math.isinf(v) else v)
                       for k, v in scores.items()}
        out_path.write_text(
            json.dumps({"scores": safe_scores, "results": []}, indent=2),
            encoding="utf-8",
        )
        loaded = json.loads(out_path.read_text())
        check("JSON write + read round-trip", True, str(safe_scores))
    except Exception as e:
        check("JSON write + read round-trip", False, str(e))

try:
    # Constrain retries so a dead server fails in <5 s instead of ~2 min.
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"]     = "5"
    import mlflow
    mlflow.set_tracking_uri("http://localhost:9999")   # intentionally wrong port
    _t_mlflow = time.perf_counter()
    try:
        mlflow.set_experiment("rbi-intel-dry-run")
        with mlflow.start_run(run_name="dry_run"):
            mlflow.log_metrics({"test": 1.0})
        log.warning("MLflow connected on port 9999 -- check if that is expected")
    except Exception:
        pass  # expected: connection refused
    elapsed_mlflow = time.perf_counter() - _t_mlflow
    check("MLflow fails gracefully (no crash)", True,
          f"failed in {elapsed_mlflow:.1f}s (expected < 10s)")
    check("MLflow fast fail (< 10 s)", elapsed_mlflow < 10,
          f"took {elapsed_mlflow:.1f}s -- retries may not be constrained")
except Exception as e:
    check("MLflow fails gracefully", False, str(e))


# ---------------------------------------------------------------------------
# Phase 4 -- InferenceLogger write
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
log.info("Phase 4: InferenceLogger logs/inference.jsonl write")
log.info("=" * 60)

try:
    from rag._logger import InferenceLogger
    lg = InferenceLogger("dry_run", "test question")
    lg.step("embed_retrieve", latency_s=0.05, n_chunks=3, top_score=0.81)
    lg.finish("test answer", n_contexts=3)

    log_path = Path(__file__).parent.parent / "logs" / "inference.jsonl"
    if log_path.exists():
        lines = [l for l in log_path.read_text().split("\n") if l.strip()]
        last  = json.loads(lines[-1])
        ok    = last.get("stage") == "dry_run"
        check("InferenceLogger wrote to logs/inference.jsonl", ok,
              f"last stage={last.get('stage')!r}")
    else:
        check("logs/inference.jsonl exists after write", False, str(log_path))
except Exception as e:
    check("InferenceLogger end-to-end", False, str(e))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log.info("")
log.info("=" * 60)
if not _failures:
    log.info("ALL CHECKS PASSED -- safe to run the full eval")
    log.info("Next: ./venv/bin/python eval/ragas_eval.py --stage naive")
    sys.exit(0)
else:
    log.error("FAILED CHECKS (%d): %s", len(_failures), _failures)
    log.error("Fix these before running ragas_eval.py")
    sys.exit(1)
