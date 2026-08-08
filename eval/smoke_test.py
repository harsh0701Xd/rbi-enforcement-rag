"""
eval/smoke_test.py

CI smoke test gate — 10 stratified questions, Stage 3 only.

Gate thresholds (config.yaml evaluation section):
  faithfulness    >= 0.65  (faithfulness_gate)
  answer_relevancy >= 0.60  (relevancy_gate)

Exits 0 on pass, exits 1 on fail (consumed by GitHub Actions deploy workflow).
Uses Claude Sonnet as RAGAS judge — NEVER Haiku (circular evaluation bias).

Stratified sample: 5 Tier 1, 3 Tier 2, 2 Tier 3 (10 total).

Usage:
    python eval/smoke_test.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from config import get_config
from eval._runner import run_stage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

_GT_FILE = Path(__file__).parent / "ground_truth.json"


def _stratified_sample(pairs: list[dict], n: int) -> list[dict]:
    """
    Pull n questions with fixed tier allocation: 5×T1, 3×T2, 2×T3.
    If a tier has fewer questions than allocated, takes all available.
    """
    by_tier: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for p in pairs:
        by_tier[p.get("tier", 1)].append(p)

    alloc   = {1: 5, 2: 3, 3: 2}
    sample: list[dict] = []
    for tier, count in alloc.items():
        sample.extend(by_tier[tier][:count])
    return sample[:n]


def main() -> None:
    cfg          = get_config()
    faith_gate   = cfg["evaluation"]["faithfulness_gate"]
    rel_gate     = cfg["evaluation"]["relevancy_gate"]
    n            = cfg["evaluation"]["smoke_test_count"]
    judge_model  = cfg["models"]["judge_model"]   # Sonnet — never change to Haiku

    # Load stratified sample
    data   = json.loads(_GT_FILE.read_text(encoding="utf-8"))
    sample = _stratified_sample(data["qa_pairs"], n)
    t_counts = {t: sum(1 for p in sample if p.get("tier") == t) for t in (1, 2, 3)}
    log.info("Smoke test: %d questions | T1=%d T2=%d T3=%d",
             len(sample), t_counts[1], t_counts[2], t_counts[3])

    # Configure RAGAS judge (Sonnet) + embeddings
    llm  = LangchainLLMWrapper(ChatAnthropic(model=judge_model, temperature=0))
    emb  = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    faithfulness.llm     = llm
    answer_relevancy.llm = llm
    if hasattr(answer_relevancy, "embeddings"):
        answer_relevancy.embeddings = emb

    # Run Stage 3 (the deployed stage)
    results = run_stage("agentic", sample, cfg)
    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    scores      = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    faith_score = float(scores["faithfulness"])
    rel_score   = float(scores["answer_relevancy"])

    print("\n── Smoke Test Results ──────────────────────────")
    print(f"  faithfulness     : {faith_score:.4f}  (gate >= {faith_gate})"
          f"  {'PASS' if faith_score >= faith_gate else 'FAIL'}")
    print(f"  answer_relevancy : {rel_score:.4f}  (gate >= {rel_gate})"
          f"  {'PASS' if rel_score >= rel_gate else 'FAIL'}")

    passed = faith_score >= faith_gate and rel_score >= rel_gate
    if passed:
        log.info("SMOKE TEST PASSED -- deployment gate met")
        sys.exit(0)
    else:
        log.error("SMOKE TEST FAILED -- deployment gate not met")
        sys.exit(1)


if __name__ == "__main__":
    main()
