"""
query_cli.py

Interactive query CLI for manual pre-deployment validation of Stage 4 Agentic RAG.

Usage:
    venv/bin/python3 query_cli.py

Commands at the prompt:
    <question>   -- run question through Stage 4 pipeline
    quit / exit  -- exit the CLI
    ---          -- print a separator between sessions
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Suppress noisy library logs; keep our own INFO
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s  %(message)s",
)
logging.getLogger("rag").setLevel(logging.WARNING)
logging.getLogger("retrieval").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

from config import get_config
from rag.agentic_rag import answer
from rag._entities import extract_entities

# ── display helpers ───────────────────────────────────────────────────────────

_W = 80  # terminal width

def _hr(char: str = "─") -> None:
    print(char * _W)

def _section(title: str) -> None:
    print(f"\n  ▸ {title}")
    print("  " + "─" * (_W - 4))

def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_W - indent, initial_indent=prefix,
                         subsequent_indent=prefix)

def _display(result: dict, question: str) -> None:
    _hr("═")
    print(f"  QUERY: {question}")
    _hr("═")

    # ── sub-questions and routes ──────────────────────────────────────────────
    _section("Decomposition & Routing")
    subs   = result.get("sub_questions", [question])
    routes = result.get("routes", [])
    for i, (sq, rt) in enumerate(zip(subs, routes), 1):
        route_badge = {"vector": "🔵 vector", "graph": "🟢 graph",
                       "both": "🟣 both"}.get(rt, rt)
        print(f"    [{i}] {route_badge}")
        print(_wrap(sq, indent=8))

    # ── filter applied ────────────────────────────────────────────────────────
    _section("Entity Pre-filter (per sub-question)")
    any_filter = False
    for sq, rt in zip(subs, routes):
        if rt in ("vector", "both"):
            entities = extract_entities(sq)
            if entities:
                print(f"    entity_name $in {entities}")
                print(_wrap(f"→ for: {sq}", indent=8))
                any_filter = True
    if not any_filter:
        print("    None — full corpus scan (no named entity detected)")

    # ── retrieved contexts ────────────────────────────────────────────────────
    contexts: list[str] = result.get("contexts", [])
    graph_rows: list[dict] = result.get("graph_rows", [])

    # Split contexts into vector vs graph-row strings (graph rows are short,
    # don't start with "Entity:")
    vector_ctxs = [c for c in contexts if c.startswith("Entity:")]
    graph_ctxs  = [c for c in contexts if not c.startswith("Entity:")]

    _section(f"Vector Chunks  ({len(vector_ctxs)} retrieved)")
    if vector_ctxs:
        for i, chunk in enumerate(vector_ctxs, 1):
            lines = chunk.split("\n", 1)
            header = lines[0] if lines else ""
            body   = lines[1].strip() if len(lines) > 1 else ""
            print(f"    [{i}] {header}")
            if body:
                print(_wrap(body[:300] + ("…" if len(body) > 300 else ""),
                            indent=8))
    else:
        print("    (none)")

    _section(f"Graph Rows     ({len(graph_rows)} raw rows → {len(graph_ctxs)} context strings)")
    if graph_ctxs:
        for i, gs in enumerate(graph_ctxs[:10], 1):   # cap display at 10
            print(f"    [{i}] {gs[:120]}{'…' if len(gs) > 120 else ''}")
        if len(graph_ctxs) > 10:
            print(f"    … and {len(graph_ctxs) - 10} more")
    else:
        print("    (none)")

    # ── final answer ──────────────────────────────────────────────────────────
    _section("Answer")
    answer_text = result.get("answer", "")
    # Wrap each paragraph separately
    for para in answer_text.split("\n"):
        if para.strip():
            print(_wrap(para, indent=4))
        else:
            print()

    _hr()
    print()


# ── main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    _hr("═")
    print("  FinGraph Intelligence System — Stage 4 Agentic RAG  (pre-deployment CLI)")
    _hr("═")
    print("  Type a question and press Enter.  Commands: quit / exit")
    print()

    cfg = get_config()

    # Warm up ChromaDB + cross-encoder on first query (avoids cold-start lag)
    warmup_done = False

    while True:
        try:
            question = input("❓ Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        if question == "---":
            _hr()
            continue

        if not warmup_done:
            print("  (loading models on first query — subsequent queries are faster…)")
            warmup_done = True

        print()
        try:
            result = answer(question, cfg=cfg)
            _display(result, question)
        except Exception as exc:
            print(f"  ✗ Error: {exc}")
            print()


if __name__ == "__main__":
    main()
