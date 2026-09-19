"""
frontend/app.py

Streamlit frontend for FinGraph Intelligence System.

Layout: side-by-side Option B
  Left  (60%): query input (chip examples + text area + Ask button) · answer · MLflow badge
  Right (40%): pipeline internals — mode+latency, hop resolution, retrieval stats,
               decomposition, Cypher query; "Show full detail" → grouped graph table +
               vector chunk expanders

Environment variables:
  API_URL     — FastAPI base URL  (default: http://localhost:8000)
  MLFLOW_URL  — MLflow UI URL     (default: http://localhost:5000)

Usage:
  API_URL=http://localhost:8001 MLFLOW_URL=http://localhost:5050 \\
      venv/bin/python3 -m streamlit run frontend/app.py
"""

from __future__ import annotations

import html as _html
import os
import re
from collections import defaultdict

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinGraph · RBI Enforcement",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL    = os.getenv("API_URL",    "http://localhost:8080")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── global resets ── */
* { box-sizing: border-box; }

/* ── top bar ── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 1rem;
}
.topbar-left { display: flex; align-items: center; gap: 10px; }
.app-title   { font-size: 1.3rem; font-weight: 500;
               color: var(--text-color, #e2e8f0) !important; margin: 0; }
.app-sub     { font-size: .8rem; color: #94a3b8 !important; margin: 0; }

/* ── chip example buttons ── */
div[data-testid="stHorizontalBlock"]:first-of-type button {
  font-size: .72rem !important; padding: 3px 8px !important;
  border-radius: 12px !important; border: 0.5px solid #475569 !important;
  color: #94a3b8 !important; background: transparent !important;
  white-space: normal !important; height: auto !important;
  min-height: unset !important; line-height: 1.3 !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type button:hover {
  border-color: #3b82f6 !important; color: #93c5fd !important;
  background: rgba(59,130,246,.08) !important;
}

/* ── answer box ── */
.answer-lbl  { font-size: .7rem; font-weight: 600; color: #64748b;
               text-transform: uppercase; letter-spacing: .06em; margin-bottom: .3rem; }
.answer-box  {
  border-left: 3px solid #3b82f6;
  border-radius: 0 8px 8px 0;
  padding: .9rem 1.1rem;
  font-size: .9rem; line-height: 1.65;
  color: #e2e8f0 !important;
  background: rgba(59,130,246,.07) !important;
  max-height: 300px; overflow-y: auto;
}

/* ── meta row (mlflow badge) ── */
.meta-row   { display: flex; align-items: center;
              justify-content: space-between; margin-top: .6rem; }
.mlf-badge  { display: inline-flex; align-items: center; gap: 5px;
              font-size: .75rem; padding: 3px 9px; border-radius: 10px;
              background: #052e16; color: #4ade80; }
.mlf-link   { font-size: .75rem; color: #60a5fa;
              text-decoration: none; margin-left: 6px; }

/* ── right panel section headers ── */
.rp-hdr {
  font-size: .7rem; font-weight: 600; color: #64748b;
  text-transform: uppercase; letter-spacing: .06em;
  margin: .85rem 0 .3rem 0;
}
.rp-hdr:first-child { margin-top: 0; }

/* ── mode badge ── */
.mode-line   { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.badge-serial   { background: #451a03; color: #fcd34d;
                  padding: 2px 9px; border-radius: 10px;
                  font-size: .78rem; font-weight: 600; }
.badge-parallel { background: #1e3a8a; color: #93c5fd;
                  padding: 2px 9px; border-radius: 10px;
                  font-size: .78rem; font-weight: 600; }
.latency    { font-size: .76rem; color: #64748b; }
.cypher-hint { font-size: .74rem; color: #64748b; margin-top: 3px; }

/* ── hop tag ── */
.hop-row  { display: flex; align-items: center; gap: 6px;
            font-size: .78rem; color: #94a3b8; margin-bottom: 3px; }
.hop-entity { background: #422006; color: #fcd34d; padding: 1px 8px;
              border-radius: 8px; font-size: .72rem;
              max-width: 200px; overflow: hidden;
              text-overflow: ellipsis; white-space: nowrap;
              display: inline-block; }

/* ── stat cards ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
             margin-bottom: .2rem; }
.stat-card { background: rgba(255,255,255,.04);
             border: 0.5px solid #334155;
             border-radius: 8px; padding: 8px 10px; text-align: center; }
.stat-num  { font-size: 1.45rem; font-weight: 500; color: #60a5fa; }
.stat-lbl  { font-size: .68rem; color: #64748b; margin-top: 1px; }

/* ── route badges ── */
.rb { font-size: .72rem; padding: 1px 7px; border-radius: 8px;
      font-weight: 600; display: inline-block; }
.rb-graph  { background: #052e16; color: #4ade80; }
.rb-vector { background: #1e3a8a; color: #93c5fd; }
.rb-both   { background: #2e1065; color: #c4b5fd; }
.dep-tag   { background: #422006; color: #fcd34d;
             font-size: .7rem; padding: 1px 7px; border-radius: 8px; }
.subq-item { margin-bottom: .5rem; }
.subq-meta { display: flex; align-items: center; gap: 5px;
             margin-bottom: 2px; }
.subq-num  { font-size: .72rem; color: #64748b; }
.subq-text { font-size: .78rem; color: #94a3b8; line-height: 1.4;
             padding-left: 2px; }

/* ── full detail panel ── */
.detail-sep { border: none; border-top: 0.5px solid #334155;
              margin: .8rem 0 .6rem 0; }

/* ── graph table ── */
.graph-tbl-wrap { overflow-x: auto; margin-top: .3rem; }
.graph-tbl {
  width: 100%; border-collapse: collapse; font-size: .76rem;
}
.graph-tbl th {
  text-align: left; color: #64748b; font-weight: 500;
  padding: 4px 8px; border-bottom: 0.5px solid #334155;
  white-space: nowrap;
}
.graph-tbl td {
  padding: 5px 8px; color: #e2e8f0;
  border-bottom: 0.5px solid #1e293b; vertical-align: top;
}
.graph-tbl tr:last-child td { border-bottom: none; }
.graph-tbl tr:hover td { background: rgba(255,255,255,.03); }
.td-null   { color: #475569; font-style: italic; }
.td-accent { color: #60a5fa; font-weight: 500; text-align: center; }
.td-mono   { font-family: monospace; font-size: .71rem; color: #64748b; }
.td-trunc  { max-width: 140px; overflow: hidden;
             text-overflow: ellipsis; white-space: nowrap; }
.td-citations { color: #94a3b8; font-size: .72rem; }

/* ── chunk expander tweak ── */
div[data-testid="stExpander"] summary {
  font-size: .78rem !important;
}

/* ── right panel separator ── */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
  border-left: 0.5px solid #1e293b;
  padding-left: 1rem;
}

/* ── hide streamlit branding ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "result"      not in st.session_state: st.session_state.result      = None
if "question"    not in st.session_state: st.session_state.question    = ""
if "error"       not in st.session_state: st.session_state.error       = None
if "query_draft" not in st.session_state: st.session_state.query_draft = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

EXAMPLES = [
    "Which bank received the highest penalty in 2023, and for what violation?",
    "Which cooperative bank was penalised the most times and what did the press release say?",
    "How many banks were penalised by RBI in 2024?",
    "Which NBFC received the largest penalty in 2022, and what regulation did it violate?",
]

# Matches both Stage 6 {q0.field_name} and legacy Stage 5 {q0_answer} syntax
_TMPL_RE = re.compile(r"\{q(\d+)(?:\.(\w+)|_answer)\}")


def _hop_display(value: "dict | str | None") -> str:
    """
    Extract a human-readable string from a hop_results value.
    Stage 6: values are raw rows[0] dicts; Stage 5 legacy: strings.
    """
    if value is None:
        return "—"
    if isinstance(value, dict):
        for key in ("entity_name", "name", "institution_name", "i.name"):
            v = value.get(key)
            if v is not None:
                return str(v)
        # Fallback: first non-null value
        for v in value.values():
            if v is not None:
                return str(v)
        return str(value)
    return str(value)


def _resolve(q: str, hop_results: dict) -> str:
    """
    Substitute template placeholders with resolved hop values.
    Handles both {q0.field_name} (Stage 6) and {q0_answer} (Stage 5 legacy).
    hop_results values may be dicts (Stage 6) or strings (legacy).
    """
    def _sub(m: re.Match) -> str:
        hop_id = m.group(1)          # "0", "1", etc.
        field  = m.group(2)          # field name for {q0.field} or None for {q0_answer}
        val    = hop_results.get(hop_id)
        if val is None:
            return m.group(0)  # leave placeholder
        if isinstance(val, dict):
            if field:
                # Try exact match then alias-stripped match
                v = val.get(field)
                if v is None:
                    for k, v2 in val.items():
                        if k.split(".")[-1] == field:
                            v = v2
                            break
                return str(v) if v is not None else m.group(0)
            else:
                return _hop_display(val)
        return str(val)
    return _TMPL_RE.sub(_sub, q)


def _annotate_template(q: str) -> str:
    """
    For cypher_sufficient=True queries: escape HTML then replace template
    placeholders with a styled inline tag.  hop_results is empty in this path
    (the single Cypher resolved all hops implicitly).
    Handles both {q0.field} and {q0_answer} syntax.
    """
    escaped = _html.escape(q)
    # Re-unescape the placeholder braces so the regex can match them
    escaped = escaped.replace("&#123;", "{").replace("&#125;", "}")
    return _TMPL_RE.sub(
        lambda m: f'<span class="dep-tag" style="font-size:.7rem;vertical-align:middle;">↳ hop {m.group(1)}</span>',
        _html.escape(q),  # start from clean escape
    )


def _query_api(question: str) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/query",
                             json={"question": question}, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.session_state.error = (
            f"Cannot connect to API at **{API_URL}**.  \n"
            "Start the server:  \n"
            "`venv/bin/python3 -m uvicorn serving.api:app --host 0.0.0.0 --port 8001`"
        )
    except requests.exceptions.Timeout:
        st.session_state.error = "Request timed out (>180 s)."
    except requests.exceptions.HTTPError as e:
        st.session_state.error = f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        st.session_state.error = f"Unexpected error: {e}"
    return None


# Stage 6: suffix-based citation column detection — more general than a fixed set
def _is_citation_col(col: str) -> bool:
    c = col.lower()
    return (
        c.endswith(("_id", "_number", "_url", "_citation"))
        or c in {"title", "source", "url", "citation", "prid"}
    )

# Kept for backward compat with _group_graph_rows / _graph_table_html internal use
_CITATION_COLS: "set[str]" = set()  # no longer used directly; replaced by _is_citation_col()

def _group_graph_rows(graph_rows: list[dict]) -> list[dict]:
    """
    Group graph rows to eliminate Cartesian-product duplicates.

    When a Cypher JOIN produces N violations × M citations, we get N×M rows.
    This groups by all non-citation columns and collapses citation columns into
    a " · "-separated string so each logical violation appears once.

    Falls back to returning rows unchanged if no grouping is needed or
    the structure is unexpected.
    """
    if not graph_rows:
        return graph_rows

    # Normalise keys (strip alias prefix)
    cleaned = [{k.split(".")[-1]: v for k, v in row.items()} for row in graph_rows]
    cols = list(cleaned[0].keys())

    citation_cols = [c for c in cols if _is_citation_col(c)]
    key_cols      = [c for c in cols if not _is_citation_col(c)]

    if not citation_cols:
        # Nothing to group — return cleaned rows as-is
        return cleaned

    # Group by key columns
    groups: dict[tuple, dict] = {}
    cit_buckets: dict[tuple, dict[str, set]] = defaultdict(lambda: {c: set() for c in citation_cols})

    def _hashable(v):
        """Convert list/dict values to hashable equivalents for dict-key use."""
        if isinstance(v, list):
            return tuple(v)
        if isinstance(v, dict):
            return tuple(sorted(v.items()))
        return v

    for row in cleaned:
        key = tuple(_hashable(row.get(c)) for c in key_cols)
        if key not in groups:
            groups[key] = {c: row.get(c) for c in key_cols}
        for cc in citation_cols:
            val = row.get(cc)
            if val is not None and str(val).strip() not in ("", "None"):
                cit_buckets[key][cc].add(str(val))

    result = []
    for key, base_row in groups.items():
        merged = dict(base_row)
        for cc in citation_cols:
            vals = sorted(cit_buckets[key][cc])
            merged[cc] = " · ".join(vals) if vals else None
        result.append(merged)

    return result


def _graph_table_html(graph_rows: list[dict]) -> str:
    """
    Render (already-grouped) graph rows as a styled HTML table.
    Null values → em dash. Aggregate numeric columns → accent blue.
    Citation columns → muted gray.
    """
    if not graph_rows:
        return ""

    cols = list(graph_rows[0].keys())
    _agg = {"count", "total", "sum", "num", "n_"}

    def _is_agg(col: str) -> bool:
        return any(tok in col.lower() for tok in _agg)

    def _is_citation(col: str) -> bool:
        return _is_citation_col(col)

    rows_html = ""
    for row in graph_rows:
        cells = ""
        for col in cols:
            val = row.get(col)
            if val is None or str(val).strip() in ("None", ""):
                cells += '<td class="td-null">—</td>'
            elif _is_agg(col):
                cells += f'<td class="td-accent">{_html.escape(str(val))}</td>'
            elif _is_citation(col):
                cells += f'<td class="td-citations">{_html.escape(str(val))}</td>'
            elif col in ("title",):
                short = str(val)[:55] + "…" if len(str(val)) > 55 else str(val)
                cells += f'<td class="td-trunc" title="{_html.escape(str(val))}">{_html.escape(short)}</td>'
            else:
                cells += f'<td>{_html.escape(str(val))}</td>'
        rows_html += f"<tr>{cells}</tr>"

    hdrs = "".join(f"<th>{_html.escape(c)}</th>" for c in cols)
    return (
        f'<div class="graph-tbl-wrap">'
        f'<table class="graph-tbl"><thead><tr>{hdrs}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>'
    )


# ── Top bar ───────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="topbar">'
    '  <div class="topbar-left">'
    '    <div>'
    '      <p class="app-title">🔍 FinGraph Intelligence System</p>'
    '      <p class="app-sub">Multi-hop agentic RAG over RBI enforcement data</p>'
    '    </div>'
    '  </div>'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────

left, right = st.columns([3, 2], gap="medium")


# ════════════════════════════════════════════════════════════
# LEFT — query + answer
# ════════════════════════════════════════════════════════════

with left:

    # Example question chips — one st.button per chip, styled as pills via CSS
    chip_cols = st.columns(len(EXAMPLES))
    for i, (col, eq) in enumerate(zip(chip_cols, EXAMPLES)):
        with col:
            label = eq[:48] + "…" if len(eq) > 48 else eq
            if st.button(label, key=f"chip_{i}", use_container_width=True):
                st.session_state.question    = eq
                st.session_state.query_draft = eq
                st.session_state.result      = None
                st.session_state.error       = None
                st.rerun()

    # Query form
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "question",
            key="query_draft",
            height=80,
            placeholder="Ask anything about RBI enforcement actions…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", use_container_width=True, type="primary")

    if submitted and question.strip():
        st.session_state.question = question.strip()
        st.session_state.error    = None
        with st.spinner("Running pipeline…"):
            st.session_state.result = _query_api(question.strip())

    if st.session_state.error:
        st.error(st.session_state.error)

    result = st.session_state.result
    if result:
        # Answer
        st.markdown('<div class="answer-lbl">Answer</div>', unsafe_allow_html=True)
        safe = _html.escape(result["answer"]).replace("\n", "<br>")
        st.markdown(f'<div class="answer-box">{safe}</div>', unsafe_allow_html=True)

        # MLflow badge (latency has moved to right panel, inline with mode badge)
        st.markdown(
            f'<div class="meta-row">'
            f'  <div>'
            f'    <span class="mlf-badge">✓ Logged to MLflow</span>'
            f'    <a class="mlf-link" href="{MLFLOW_URL}" target="_blank">Open dashboard →</a>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#475569;font-size:.85rem;margin-top:2rem;text-align:center;">'
            'Enter a question above to query the RBI enforcement corpus.'
            '</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════
# RIGHT — pipeline internals
# ════════════════════════════════════════════════════════════

with right:
    result = st.session_state.result

    if not result:
        st.markdown(
            '<div style="color:#475569;font-size:.82rem;margin-top:.5rem;">'
            'Pipeline details appear here after a query runs.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        cypher_ok = result.get("cypher_sufficient", False)
        fan_out   = result.get("fan_out", False)
        hop_res   = result.get("hop_results", {})
        sub_qs    = result.get("sub_questions", [])
        routes    = result.get("routes", [])
        hop_deps  = result.get("hop_dependencies", [None] * len(sub_qs))
        latency   = result.get("latency_s", 0.0)

        # ── Execution path badge + latency ───────────────────────────────
        st.markdown('<div class="rp-hdr">Mode</div>', unsafe_allow_html=True)

        # Stage 6: derive badge from cypher_sufficient + fan_out
        # (execution_mode is always "serial" and no longer meaningful for display)
        if fan_out:
            badge_inner = '<span class="badge-parallel">⟳ fan-out</span>'
            path_label  = "Fan-out · Q2 over all Q1 rows"
            path_color  = "#93c5fd"
        elif cypher_ok:
            badge_inner = '<span class="badge-serial">⚡ single-shot</span>'
            path_label  = "Single-shot · WITH-chained Cypher"
            path_color  = "#fcd34d"
        else:
            badge_inner = '<span class="badge-serial">⚡ serial</span>'
            path_label  = "Serial · Hybrid 2-pass"
            path_color  = "#fcd34d"

        st.markdown(
            f'<div class="mode-line">'
            f'  {badge_inner}'
            f'  <span class="latency">⏱ {latency:.1f} s</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:.73rem;color:{path_color};margin-top:4px;">'
            f'⤷ {path_label}</div>',
            unsafe_allow_html=True,
        )

        icon = "✓" if cypher_ok else "✗"
        st.markdown(
            f'<div class="cypher-hint">cypher_sufficient = {icon} {cypher_ok}'
            f'{"  ·  fan_out = ✓" if fan_out else ""}</div>',
            unsafe_allow_html=True,
        )

        # ── Hop resolution ───────────────────────────────────────────────
        # Only show when there are actual dependent hops resolved at pipeline level.
        # Hide for: single sub-question queries, cypher_sufficient=True (single Cypher
        # resolved all hops implicitly — no pipeline-level resolution occurred).
        has_dependencies = any(d is not None for d in hop_deps)
        show_hop_resolution = has_dependencies and not cypher_ok and hop_res
        if show_hop_resolution:
            st.markdown('<div class="rp-hdr">Hop resolution</div>', unsafe_allow_html=True)
            for hop_id, hop_val in hop_res.items():
                display = _hop_display(hop_val)
                st.markdown(
                    f'<div class="hop-row">'
                    f'  <span style="font-size:.8rem;">→</span>'
                    f'  <span>hop {hop_id}</span>'
                    f'  <span class="hop-entity">{_html.escape(display)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Retrieval stats ───────────────────────────────────────────────
        st.markdown('<div class="rp-hdr">Retrieval</div>', unsafe_allow_html=True)
        # cypher_sufficient=True → single Cypher, no vector retrieval
        # fan_out=True → only graph results (vector skipped in fan-out path)
        n_vec       = 0 if (cypher_ok or fan_out) else result["n_vector_chunks"]
        n_rows      = result["n_graph_rows"]
        n_attempted = result.get("n_graph_attempted", 0)
        n_succeeded = result.get("n_graph_succeeded", 0)

        st.markdown(
            f'<div class="stat-grid">'
            f'  <div class="stat-card">'
            f'    <div class="stat-num">{n_vec}</div>'
            f'    <div class="stat-lbl">vector chunks</div>'
            f'  </div>'
            f'  <div class="stat-card">'
            f'    <div class="stat-num">{n_rows}</div>'
            f'    <div class="stat-lbl">graph rows</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Fix 1: Graph retrieval success/failure indicator
        if n_attempted > 0:
            all_ok   = n_succeeded == n_attempted
            none_ok  = n_succeeded == 0
            icon     = "✓" if all_ok else ("✗" if none_ok else "⚠")
            color    = "#4ade80" if all_ok else ("#f87171" if none_ok else "#fcd34d")
            st.markdown(
                f'<div style="font-size:.73rem;color:{color};margin-top:4px;">'
                f'{icon} graph: {n_succeeded}/{n_attempted} queries succeeded</div>',
                unsafe_allow_html=True,
            )

        # ── Decomposition & routing ───────────────────────────────────────
        st.markdown('<div class="rp-hdr">Decomposition &amp; routing</div>',
                    unsafe_allow_html=True)

        _rb_map = {
            "graph":  '<span class="rb rb-graph">graph</span>',
            "vector": '<span class="rb rb-vector">vector</span>',
            "both":   '<span class="rb rb-both">both</span>',
        }

        for i, (sq, rt, dep) in enumerate(zip(sub_qs, routes, hop_deps)):
            # When cypher_sufficient=True, single-shot Cypher covers everything
            # via graph — suppress "both"/"vector" route badges (not actual execution)
            if cypher_ok:
                rt = "graph"

            rb = _rb_map.get(rt, f"<span>{rt}</span>")

            # Task #11: Suppress dep-tag for cypher_sufficient=True — the single-shot
            # Cypher covers all hops implicitly; there is no pipeline-level resolution.
            if dep is not None and not cypher_ok:
                dep_html = f' <span class="dep-tag">depends on hop {dep}</span>'
            else:
                dep_html = ""

            # Annotate template placeholders:
            # cypher_sufficient=True → styled tag (hop resolved implicitly by single Cypher)
            # otherwise → substitute real resolved value from hop_results
            if cypher_ok:
                display_q = _annotate_template(sq)
            else:
                display_q = _html.escape(_resolve(sq, hop_res))

            st.markdown(
                f'<div class="subq-item">'
                f'  <div class="subq-meta">'
                f'    <span class="subq-num">Q{i+1}</span>{rb}{dep_html}'
                f'  </div>'
                f'  <div class="subq-text">{display_q}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Full detail toggle ────────────────────────────────────────────
        st.markdown('<hr class="detail-sep">', unsafe_allow_html=True)
        show_full = st.toggle("Show full detail", value=False, key="full_detail")

        if show_full:
            contexts       = result.get("contexts", [])
            graph_rows_raw = result.get("graph_rows", [])
            cypher_queries = result.get("cypher_queries", [])
            vec_chunks     = [c for c in contexts if c.startswith("Entity:")]

            # Per-hop Cypher display with label + success indicator
            if cypher_queries:
                st.markdown('<div class="rp-hdr">Cypher queries</div>',
                            unsafe_allow_html=True)
                for cq_idx, cq in enumerate(cypher_queries):
                    if isinstance(cq, dict):
                        # Task #9: for single-shot path, override "Q1" label to show
                        # all covered sub-questions so it's clear one Cypher covered them all
                        label = cq.get("label", "Q?")
                        if cypher_ok and len(sub_qs) > 1 and cq_idx == 0:
                            covered = ", ".join(f"Q{j+1}" for j in range(len(sub_qs)))
                            label   = f"Single-shot ({covered})"

                        success = cq.get("success", True)
                        n_r     = cq.get("n_rows", 0)
                        icon    = "✓" if success else "✗"
                        color   = "#4ade80" if success else "#f87171"
                        st.markdown(
                            f'<div style="font-size:.73rem;color:{color};'
                            f'margin:.5rem 0 .15rem;">'
                            f'{icon} {label} · {n_r} rows</div>',
                            unsafe_allow_html=True,
                        )
                        st.code(cq.get("cypher", ""), language="cypher")
                    else:
                        # backward-compat: plain string (old API response)
                        st.code(cq, language="cypher")

            # Fix #6: Graph rows with grouping to collapse Cartesian-product duplicates
            if graph_rows_raw:
                grouped = _group_graph_rows(graph_rows_raw)
                st.markdown(
                    f'<div class="rp-hdr">Graph rows'
                    f'<span style="font-weight:400;color:#475569;"> '
                    f'({len(grouped)} grouped / {len(graph_rows_raw)} raw)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                tbl_html = _graph_table_html(grouped[:20])
                st.markdown(tbl_html, unsafe_allow_html=True)
                if len(grouped) > 20:
                    st.caption(f"Showing first 20 of {len(grouped)} grouped rows")
            else:
                st.markdown(
                    '<span style="font-size:.78rem;color:#475569;">No graph rows</span>',
                    unsafe_allow_html=True,
                )

            # Vector chunks as collapsible expanders
            if vec_chunks:
                st.markdown('<div class="rp-hdr">Vector chunks</div>',
                            unsafe_allow_html=True)
                for i, chunk in enumerate(vec_chunks, 1):
                    lines  = chunk.split("\n", 1)
                    header = lines[0]  # "Entity: X | Date: Y | PR: Z"
                    body   = lines[1].strip() if len(lines) > 1 else ""
                    if len(body) > 400:
                        body = body[:400] + "…"
                    with st.expander(f"V{i} · {header[:65]}", expanded=(i == 1)):
                        st.caption(body)
            elif cypher_ok:
                # cypher_sufficient=True → single WITH-chained Cypher resolved
                # the full query; vector retrieval was intentionally skipped.
                st.markdown(
                    '<div class="rp-hdr">Vector chunks</div>'
                    '<span style="font-size:.78rem;color:#475569;font-style:italic;">'
                    'Skipped — single-shot Cypher covered this query entirely.</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="rp-hdr">Vector chunks</div>'
                    '<span style="font-size:.78rem;color:#475569;">No vector chunks retrieved.</span>',
                    unsafe_allow_html=True,
                )
