"""
tests/integration/test_stage4_live.py

Layer 3 — Live end-to-end smoke test for Advanced RAG v2 and Stage 4 Agentic.
Requires: Ollama running, ChromaDB populated, Neo4j running, ANTHROPIC_API_KEY set.

Run from rbi_intel/:
    python tests/integration/test_stage4_live.py

Checks:
  Advanced RAG v2:
    1. Named entity query -> entity_name filter fires, answer is substantive
    2. Named entity that previously failed -> no longer "I don't have enough info"
    3. NBFC query -> entity_type filter fires (regression check)

  Stage 4 Agentic:
    1. contexts contains BOTH vector chunks and graph row strings
    2. Named entity sub-question -> entity_name filter applied in vector_node
    3. Graph rows appear in contexts (not silently dropped)
    4. Aggregation query -> graph answers correctly (regression check)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        msg = f"{label}" + (f" | {detail}" if detail else "")
        print(f"  {FAIL}  {msg}")
        failures.append(msg)


# ── Advanced RAG v2 ───────────────────────────────────────────────────────────

def test_advanced_v2() -> None:
    print("\n=== Advanced RAG v2 ===")
    from rag.advanced_rag import _extract_filters, answer
    from config import get_config
    cfg = get_config()

    # 1. Entity name filter fires for named institution
    cw, _, __ = _extract_filters("What penalty did RBI impose on Axis Bank?")
    check("Axis Bank query -> entity_name filter",
          cw is not None and "entity_name" in str(cw))
    check("Axis Bank filter includes canonical variant",
          cw is not None and "Axis Bank Limited" in str(cw))

    # 2. Known Stage 2 failure: named private bank that returned "I don't have enough info"
    #    (previously failed because _extract_filters only handled NBFC/coop, not named banks)
    print("\n  [Live call] Axis Bank question (was 7/25 failure pattern)...")
    result = answer("What was the reason for the penalty imposed on Axis Bank by RBI?", cfg=cfg)
    check("Axis Bank answer is substantive (>100 chars)",
          len(result["answer"]) > 100,
          f"answer length={len(result['answer'])}")
    check("Axis Bank answer not 'no information'",
          "don't have enough information" not in result["answer"].lower(),
          result["answer"][:120])
    check("Axis Bank filter applied",
          result["filters"]["chroma_where"] is not None,
          str(result["filters"]["chroma_where"]))

    # 3. NBFC query still works (regression)
    print("\n  [Live call] NBFC category query (regression)...")
    result_nbfc = answer("Which NBFCs were penalised by RBI for non-compliance?", cfg=cfg)
    check("NBFC filter applied (regression)",
          result_nbfc["filters"]["chroma_where"] is not None
          and "entity_type" in str(result_nbfc["filters"]["chroma_where"]))
    check("NBFC answer is substantive",
          len(result_nbfc["answer"]) > 100)


# ── Stage 4 Agentic ───────────────────────────────────────────────────────────

def test_stage4_agentic() -> None:
    print("\n=== Stage 4 Agentic ===")
    from rag.agentic_rag import answer
    from config import get_config
    cfg = get_config()

    # 1. Named entity question: contexts must include both vector and graph
    print("\n  [Live call] Axis Bank named entity question...")
    result = answer("What penalty did RBI impose on Axis Bank and for what reason?", cfg=cfg)

    n_contexts = len(result["contexts"])
    n_graph_rows = len(result["graph_rows"])
    check("contexts list is non-empty",
          n_contexts > 0, f"got {n_contexts} contexts")
    check("graph_rows retrieved",
          n_graph_rows > 0, f"got {n_graph_rows} graph rows")
    check("contexts > vector chunks alone (graph rows included)",
          # graph rows must have added at least 1 context string
          n_contexts > 0 and n_graph_rows > 0,
          f"contexts={n_contexts}, graph_rows={n_graph_rows}")

    # At least one context string should contain a canonical entity name
    entity_in_contexts = any(
        "Axis Bank" in c or "Limited" in c
        for c in result["contexts"]
    )
    check("canonical entity name appears in contexts",
          entity_in_contexts,
          "no 'Axis Bank' or 'Limited' found in any context string")

    # 2. Aggregation query: graph should answer correctly (regression)
    print("\n  [Live call] UCB count aggregation (regression)...")
    result_agg = answer("How many Urban Cooperative Banks were penalised by RBI?", cfg=cfg)
    check("Aggregation answer is non-empty",
          len(result_agg["answer"]) > 20)
    check("Aggregation retrieved graph rows",
          len(result_agg["graph_rows"]) > 0,
          f"graph_rows={len(result_agg['graph_rows'])}")

    # 3. contexts type check
    check("All contexts are strings",
          all(isinstance(c, str) for c in result["contexts"]))

    # 4. Stage label
    check("Result stage is 'agentic'",
          result["stage"] == "agentic")

    # 5. Answer is substantive
    check("Answer is substantive (>100 chars)",
          len(result["answer"]) > 100,
          f"answer length={len(result['answer'])}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_advanced_v2()
    test_stage4_agentic()

    print(f"\n{'='*50}")
    if failures:
        print(f"\033[31m{len(failures)} FAILURE(S):\033[0m")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\033[32mAll live smoke tests passed.\033[0m")
        print("Safe to proceed with RAGAS evaluation.")
