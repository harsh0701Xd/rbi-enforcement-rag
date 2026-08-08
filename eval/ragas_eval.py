"""
eval/ragas_eval.py

RAGAS evaluation across Stage 1 (naive), Stage 2 (advanced), Stage 3 (agentic).

Runs all questions in eval/ground_truth.json through each stage, computes:
  faithfulness          -- are answer claims grounded in retrieved context?
  answer_relevancy      -- is the answer on-topic for the question?
  context_precision     -- are retrieved passages relevant? (uses ground truth)
  context_recall        -- do passages cover the ground truth? (uses ground truth)
  answer_correctness    -- factual + semantic match against ground truth
  context_entity_recall -- do retrieved passages contain key entities from ground truth?

Judge LLM:  Claude Sonnet (config: models.judge_model) -- NEVER Haiku.
Embeddings: all-MiniLM-L6-v2 via sentence-transformers (no API cost).
Scores saved to eval/results/<stage>_tier<N>.json.
MLflow logging is best-effort: skipped gracefully if server is unreachable.

Run eval/dry_run.py first to validate the full pipeline cheaply.

Usage:
    python eval/ragas_eval.py --stage all       # all 3 stages
    python eval/ragas_eval.py --stage naive
    python eval/ragas_eval.py --stage naive --tier 1
    python eval/ragas_eval.py --stage agentic --tier 3
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure rbi_intel/ is on sys.path when the script is run as eval/ragas_eval.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import math
import os

import mlflow
from datasets import Dataset
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
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

from config import get_config
from eval._runner import RunResult, run_stage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

_GT_FILE     = Path(__file__).parent / "ground_truth.json"
_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)
_ALL_STAGES  = ("naive", "advanced", "agentic")
_METRIC_KEYS = frozenset({
    "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    "answer_correctness", "context_entity_recall",
})

_ALL_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    context_entity_recall,
]


def _load_gt(tier: int | None) -> list[dict]:
    data  = json.loads(_GT_FILE.read_text(encoding="utf-8"))
    pairs = data["qa_pairs"]
    if tier is not None:
        pairs = [p for p in pairs if p.get("tier") == tier]
    log.info("Loaded %d QA pairs (tier=%s)", len(pairs), tier or "all")
    return pairs


def _setup_ragas(cfg: dict) -> None:
    """Wire Claude Sonnet + MiniLM into every RAGAS metric. Never Haiku."""
    judge_model = cfg["models"]["judge_model"]
    llm = LangchainLLMWrapper(ChatAnthropic(model=judge_model, temperature=0))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    for m in _ALL_METRICS:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = emb

    log.info("RAGAS judge: %s | embeddings: all-MiniLM-L6-v2", judge_model)
    log.info("Active metrics (%d): %s", len(_ALL_METRICS), [m.name for m in _ALL_METRICS])


def _to_dataset(results: list[RunResult]) -> Dataset:
    return Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })


def _log_mlflow(stage: str, scores: dict, cfg: dict, tier: int | None) -> None:
    """Best-effort MLflow logging -- silently skipped if server is unreachable."""
    try:
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        run_name = f"ragas_{stage}_tier{tier or 'all'}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "stage":            stage,
                "tier_filter":      tier or "all",
                "system_model":     cfg["models"]["system_model"],
                "judge_model":      cfg["models"]["judge_model"],
                "chunk_size":       cfg["ingestion"]["chunk_size"],
                "chunk_overlap":    cfg["ingestion"]["chunk_overlap"],
                "naive_top_k":      cfg["naive_rag"]["top_k"],
                "hyde_temperature": cfg["llm"]["hyde_temperature"],
                "top_k_initial":    cfg["advanced_rag"]["top_k_initial"],
                "top_k_final":      cfg["advanced_rag"]["top_k_final"],
            })
            mlflow.log_metrics({f"{stage}_{k}": v for k, v in scores.items()})
        log.info("MLflow logged: %s | %s", run_name, scores)
    except Exception as exc:
        log.warning("MLflow logging skipped (%s) -- scores still saved to JSON", exc)


def evaluate_stage(
    stage: str, qa_pairs: list[dict], cfg: dict, tier: int | None,
    suffix: str | None = None,
) -> dict[str, float]:
    """Run one stage through RAGAS; save raw results; return metric scores.

    Args:
        suffix: Optional label appended to the output filename so new runs
                don't overwrite existing baselines.
                  None  -> advanced_tierall.json   (original behaviour)
                  "v2"  -> advanced_v2_tierall.json
    """
    log.info("Running %d questions through stage=%s ...", len(qa_pairs), stage)
    results  = run_stage(stage, qa_pairs, cfg)
    dataset  = _to_dataset(results)

    log.info("Running RAGAS evaluate() on %d rows ...", len(results))
    eval_out = evaluate(dataset, metrics=_ALL_METRICS)

    # Keep only the target metric keys.
    # Replace NaN/Inf with 0.0 so json.dumps produces valid JSON on all platforms.
    scores: dict[str, float] = {}
    for k, v in eval_out.items():
        if k not in _METRIC_KEYS:
            continue
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            log.warning("Metric %s returned non-finite for stage=%s -- recorded as 0.0", k, stage)
            fv = 0.0
        scores[k] = fv

    _log_mlflow(stage, scores, cfg, tier)

    tier_tag  = f"tier{tier}" if tier else "tierall"
    suffix_tag = f"_{suffix}" if suffix else ""
    out_path  = _RESULTS_DIR / f"{stage}{suffix_tag}_{tier_tag}.json"
    out_path.write_text(
        json.dumps({"scores": scores, "results": [dict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    log.info("Saved: %s", out_path)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for FinGraph.")
    parser.add_argument("--stage", default="all",
                        choices=["all", "naive", "advanced", "agentic"])
    parser.add_argument("--tier", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument(
        "--suffix", default=None,
        help=(
            "Label appended to the output filename to avoid overwriting baselines. "
            "E.g. --suffix v2 writes advanced_v2_tierall.json instead of advanced_tierall.json."
        ),
    )
    args = parser.parse_args()

    cfg = get_config()

    # MLflow setup -- best-effort, will not crash if server is absent
    try:
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("rbi-intel-ragas-eval")
    except Exception as exc:
        log.warning("MLflow setup skipped: %s", exc)

    _setup_ragas(cfg)
    pairs  = _load_gt(args.tier)
    stages = _ALL_STAGES if args.stage == "all" else (args.stage,)

    all_scores: dict[str, dict] = {}
    for stage in stages:
        log.info("=== Stage: %s ===", stage.upper())
        all_scores[stage] = evaluate_stage(stage, pairs, cfg, args.tier, suffix=args.suffix)
        print(f"\n-- {stage.upper()} --")
        for metric, score in all_scores[stage].items():
            print(f"  {metric:<28}: {score:.4f}")

    if len(stages) > 1:
        metrics = list(next(iter(all_scores.values())).keys())
        print(f"\n-- Stage Comparison (tier={args.tier or 'all'}) --")
        header = f"  {'Metric':<28}" + "".join(f"  {s:>10}" for s in stages)
        print(header)
        for m in metrics:
            row = "".join(f"  {all_scores[s].get(m, 0):>10.4f}" for s in stages)
            print(f"  {m:<28}{row}")


if __name__ == "__main__":
    main()
