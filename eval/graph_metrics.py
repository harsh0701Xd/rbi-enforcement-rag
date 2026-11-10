"""
eval/graph_metrics.py

Custom graph-specific evaluation metrics for Stage 3 Agentic GraphRAG.
Complements RAGAS (which is text-only) with graph-aware measurements.

Metrics:
  cypher_validity_rate  -- fraction of NL->Cypher queries that executed
                           without error and returned >= 1 row
  routing_accuracy      -- fraction of questions correctly routed by the
                           Stage 3 keyword router (Tier 1->vector, 2->graph, 3->both)
  node_precision        -- fraction of returned graph rows whose values
                           appear in the ground truth answer

All metrics logged to MLflow. Targets read from config.yaml evaluation section.

Usage:
    python eval/graph_metrics.py           # all tiers
    python eval/graph_metrics.py --tier 2  # relational tier only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from config import get_config
from retrieval.graph_retriever import retrieve as graph_retrieve

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

_GT_FILE = Path(__file__).parent / "ground_truth.json"

# Proxy: expected route per tier (avoids modifying the locked ground_truth.json)
_TIER_ROUTE = {1: "vector", 2: "graph", 3: "both"}


def cypher_validity_rate(pairs: list[dict], cfg: dict) -> float:
    """Fraction of questions where graph_retriever returned success=True."""
    hits = sum(1 for p in pairs if graph_retrieve(p["question"], cfg=cfg)["success"])
    rate = hits / len(pairs) if pairs else 0.0
    log.info("cypher_validity_rate: %.3f (%d/%d)", rate, hits, len(pairs))
    return rate


def routing_accuracy(pairs: list[dict], cfg: dict) -> float:
    """
    Fraction of questions where the Stage 6 LLM decomposer assigned the
    expected primary route (proxied from tier label: 1→vector, 2→graph, 3→both).

    Stage 6 uses an LLM-based decomposer (not the Stage 3 keyword classifier),
    so this function invokes the decomposer node on each question and compares
    routes[0] against the expected route derived from the tier label.
    """
    from rag._state import FinGraphState
    from rag.nodes.decomposer import decompose

    correct = 0
    for p in pairs:
        expected = _TIER_ROUTE.get(p.get("tier", 1), "vector")
        # Build a minimal FinGraphState — decomposer only reads "question" and "iterations"
        state: FinGraphState = {
            "question":          p["question"],
            "sub_questions":     [],
            "routes":            [],
            "vector_chunks":     [],
            "graph_results":     [],
            "context":           "",
            "answer":            "",
            "iterations":        0,
            "execution_mode":    "serial",
            "cypher_sufficient": False,
            "hop_dependencies":  [],
            "hop_results":       {},
            "fan_out":           False,
        }
        try:
            result  = decompose(state)
            routes  = result.get("routes", [])
            actual  = routes[0] if routes else "graph"
        except Exception as exc:
            log.warning("Decomposer failed for '%s': %s", p["question"][:60], exc)
            actual = "graph"  # safe fallback

        if actual == expected:
            correct += 1
        else:
            log.debug("Route mismatch | expected=%-6s actual=%-6s | %s",
                      expected, actual, p["question"])
    rate = correct / len(pairs) if pairs else 0.0
    log.info("routing_accuracy (Stage 6 decomposer): %.3f (%d/%d)", rate, correct, len(pairs))
    return rate


def node_precision(pairs: list[dict], cfg: dict) -> float:
    """
    Fraction of returned graph rows where any value string-matches the
    ground truth answer (case-insensitive substring).

    Approximation — measures whether the graph retrieves relevant entities,
    not whether the answer is correct.
    """
    total = 0
    hits  = 0
    for p in pairs:
        result = graph_retrieve(p["question"], cfg=cfg)
        if not result["success"]:
            continue
        gt = p["ground_truth"].lower()
        for row in result["rows"]:
            total += 1
            if any(
                str(v).lower() in gt
                for v in row.values()
                if v is not None and not isinstance(v, bool)
            ):
                hits += 1
    rate = hits / total if total else 0.0
    log.info("node_precision: %.3f (%d/%d rows relevant)", rate, hits, total)
    return rate


def compute_all(tier: int | None = None) -> dict[str, float]:
    """Compute all three graph metrics and log to MLflow."""
    cfg   = get_config()
    data  = json.loads(_GT_FILE.read_text(encoding="utf-8"))
    pairs = data["qa_pairs"]
    if tier is not None:
        pairs = [p for p in pairs if p.get("tier") == tier]
    log.info("Running graph metrics on %d QA pairs (tier=%s)", len(pairs), tier or "all")

    scores = {
        "cypher_validity_rate": cypher_validity_rate(pairs, cfg),
        "routing_accuracy":     routing_accuracy(pairs, cfg),
        "node_precision":       node_precision(pairs, cfg),
    }

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("rbi-intel-graph-metrics")
    with mlflow.start_run(run_name=f"graph_metrics_tier{tier or 'all'}"):
        mlflow.log_params({
            "tier_filter":  tier or "all",
            "system_model": cfg["models"]["system_model"],
            "cypher_limit": cfg["agentic_rag"]["cypher_limit"],
        })
        mlflow.log_metrics(scores)

    print("\n── Graph Metrics ──────────────────────────────")
    for k, v in scores.items():
        target = cfg["evaluation"].get(f"{k}_target", "N/A")
        ok     = isinstance(target, float) and v >= target
        status = "OK" if ok else "below target"
        print(f"  {k:<26}: {v:.4f}  (target={target})  {status}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=None, choices=[1, 2, 3])
    compute_all(tier=parser.parse_args().tier)
