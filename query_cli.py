"""
query_cli.py

Interactive query CLI for manual pre-deployment validation of Stage 5 Agentic RAG.

Usage:
    venv/bin/python3 query_cli.py           # default: pipeline logs suppressed
    venv/bin/python3 query_cli.py --debug   # show INFO logs from rag/retrieval

Commands at the prompt:
    <question>   -- run question through Stage 5 pipeline
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

# --debug flag: show INFO logs from rag/retrieval for Stage 5 validation
_DEBUG_MODE = "--debug" in sys.argv

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
_pipeline_log_level = logging.INFO if _DEBUG_MODE else logging.WARNING
logging.getLogger("rag").setLevel(_pipeline_log_level)
logging.getLogger("retrieval").setLevel(_pipeline_log_level)
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

    # ── Stage 5 execution mode ────────────────────────────────────────────────
    exec_mode  = result.get("execution_mode", "parallel")
    cypher_ok  = result.get("cypher_sufficient", False)
    hop_res    = result.get("hop_results", {})
    mode_badge = "⚡ serial (Stage 5)" if exec_mode == "serial" else "⟳  parallel (Stage 4)"
    print(f"\n  ▸ Execution Mode: {mode_badge}")
    if exec_mode == "serial":
        print(f"    cypher_sufficient={cypher_ok}")
        if hop_res:
            for hop_id, entity in hop_res.items():
                print(f"    hop {hop_id} → resolved: \"{entity}\"")

    # ── sub-questions and routes ──────────────────────────────────────────────
    _section("Decomposition & Routing")
    subs   = result.get("sub_questions", [question])
    routes = result.get("routes", [])
    deps   = result.get("hop_dependencies", [None] * len(subs))
    for i, (sq, rt, dep) in enumerate(zip(subs, routes, deps), 1):
        route_badge = {"vector": "🔵 vector", "graph": "🟢 graph",
                       "both": "🟣 both"}.get(rt, rt)
        dep_str = f"  [depends on hop {dep}]" if dep is not None else ""
        print(f"    [{i}] {route_badge}{dep_str}")
        print(_wrap(sq, indent=8))

    # ── filter applied ────────────────────────────────────────────────────────
    _section("Entity Pre-filter (per sub-question)")
    any_filter = False
    for i, (sq, rt, dep) in enumerate(zip(subs, routes, deps)):
        if rt not in ("vector", "both"):
            continue
        # For dependent hops, show the resolved sub-question if available
        resolved_sq = sq
        if dep is not None and dep in hop_res:
            from rag._resolver import resolve_template
            resolved_sq = resolve_template(sq, hop_res)

        entities = extract_entities(resolved_sq)
        if entities:
            label = f"hop {i}" + (f" (resolved from \"{sq[:40]}\")" if resolved_sq != sq else "")
            print(f"    {label}:")
            print(f"    entity_name $in {entities}")
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
    mode_hint = "  [--debug for pipeline logs]" if not _DEBUG_MODE else "  [DEBUG MODE]"
    print(f"  FinGraph Intelligence System — Stage 5 Agentic RAG  (pre-deployment CLI){mode_hint}")
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
