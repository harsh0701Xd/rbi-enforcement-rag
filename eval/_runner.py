"""
eval/_runner.py

Stage execution helper shared by ragas_eval.py and smoke_test.py.
Runs a named RAG stage against a list of ground truth QA pairs and returns
structured RunResult objects ready for RAGAS Dataset construction.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from config import get_config

log = logging.getLogger(__name__)


class RunResult(TypedDict):
    question:     str
    answer:       str
    contexts:     list[str]  # text passages for RAGAS context metrics
    ground_truth: str        # reference answer from ground_truth.json
    stage:        str        # "naive" | "advanced" | "agentic"
    tier:         int        # 1 | 2 | 3


def _graph_rows_to_text(rows: list[dict]) -> list[str]:
    """
    Convert Neo4j row dicts to readable text passages for RAGAS.
    Strips Cypher alias prefixes: 'i.name: Axis Bank' -> 'name: Axis Bank'
    """
    passages: list[str] = []
    for row in rows:
        parts = [
            f"{k.split('.')[-1]}: {v}"
            for k, v in row.items()
            if v is not None
        ]
        text = ", ".join(parts)
        if text:
            passages.append(text)
    return passages


def run_stage(
    stage: str,
    qa_pairs: list[dict],
    cfg: dict | None = None,
) -> list[RunResult]:
    """
    Run all qa_pairs through the named stage and return RunResult list.

    Args:
        stage:    "naive" | "advanced" | "agentic"
        qa_pairs: list of dicts from ground_truth.json with keys
                  "question", "ground_truth", "tier"
        cfg:      Config dict. Loaded automatically if None.

    Returns:
        list[RunResult] — one per question, answer and contexts populated.
    """
    if cfg is None:
        cfg = get_config()

    if stage == "naive":
        from rag.naive_rag import answer as _answer
    elif stage == "advanced":
        from rag.advanced_rag import answer as _answer
    elif stage == "agentic":
        from rag.agentic_rag import answer as _answer
    else:
        raise ValueError(f"Unknown stage {stage!r}. Must be naive | advanced | agentic")

    results: list[RunResult] = []
    for i, pair in enumerate(qa_pairs, 1):
        q = pair["question"]
        log.info("[%d/%d] stage=%-8s | %s", i, len(qa_pairs), stage, q[:70])

        out      = _answer(q, cfg=cfg)
        contexts = list(out["contexts"])
        # Note: graph rows are now merged into AgenticResult.contexts directly
        # by agentic_rag._graph_rows_to_context_strings() (B2 change).
        # The previous manual extend here has been removed to avoid double-adding.

        results.append(RunResult(
            question=q,
            answer=out["answer"],
            contexts=contexts or ["(no context retrieved)"],
            ground_truth=pair["ground_truth"],
            stage=stage,
            tier=pair.get("tier", 1),
        ))

    log.info("Stage %-8s complete: %d results", stage, len(results))
    return results
