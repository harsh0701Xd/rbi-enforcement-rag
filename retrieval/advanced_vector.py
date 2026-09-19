"""
retrieval/advanced_vector.py

Reusable advanced vector retrieval pipeline for Stage 6 (and beyond).

Combines the Stage 2 improvements (from advanced_rag.py) into a single
callable that can be imported by hop_executor for all vector hops.

Pipeline:
  1. Metadata filter extraction  — entity name/type + date range
  2. HyDE                        — generate hypothetical RBI press release,
                                   embed that instead of the raw query
  3. Wide cosine retrieval        — ChromaDB with pre-filter, top_k_initial
  4. Python date post-filter      — ISO string lexicographic comparison
  5. Cross-encoder reranking      — top_k_vector_final chunks returned

filter_q parameter:
  Some callers (hop_executor Pass 2) pass a pre-resolution template question
  for filter extraction to prevent entity contamination — the resolved question
  may contain an entity name that should not restrict the ChromaDB filter.
  The HyDE hypothesis and cosine search always use the resolved question.

Public API:
    retrieve_advanced(question, cfg, *, filter_q=None) -> list[VectorResult]
"""

from __future__ import annotations

import logging
import re

import anthropic
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_config
from rag._entities import COOP_ENTITY_TYPES, NBFC_ENTITY_TYPES, extract_entities
from rag._reranker import rerank
from rag._synthesis import get_client as _get_client
from prompts.hyde import TEMPLATE as _HYDE_TEMPLATE
from retrieval.vector_retriever import VectorResult, retrieve as vector_retrieve

load_dotenv()

log = logging.getLogger(__name__)


# ── Regex helpers (mirrors advanced_rag.py — single source kept there) ────────

_YEAR_RANGE_RE  = re.compile(r'\b(20\d{2})\s+to\s+(20\d{2})\b', re.IGNORECASE)
_SINGLE_YEAR_RE = re.compile(r'\bin\s+(20\d{2})\b|\bduring\s+(20\d{2})\b|\b(20\d{2})\b')
_NBFC_RE        = re.compile(r'\bNBFCs?\b|\bnon.?banking financial\b', re.IGNORECASE)
_COOP_BANK_RE   = re.compile(r'\bco.?operative\s+bank\b|\burban\s+co.?op\b', re.IGNORECASE)


# ── Filter extraction ─────────────────────────────────────────────────────────

def _extract_filters(question: str) -> tuple[dict | None, str | None, str | None]:
    """
    Extract ChromaDB pre-filter and date bounds from a question.

    Returns:
        chroma_where : ChromaDB where clause or None.
        date_from    : "YYYY-MM-DD" inclusive lower bound or None.
        date_to      : "YYYY-MM-DD" exclusive upper bound or None.
    """
    chroma_conds: list[dict] = []
    date_from: str | None = None
    date_to:   str | None = None

    # Date range — post-filtered in Python (date field is ISO string, not numeric)
    range_m = _YEAR_RANGE_RE.search(question)
    if range_m:
        y1, y2 = int(range_m.group(1)), int(range_m.group(2))
        lo, hi = min(y1, y2), max(y1, y2)
        date_from, date_to = f"{lo}-01-01", f"{hi + 1}-01-01"
    else:
        year_m = _SINGLE_YEAR_RE.search(question)
        if year_m:
            y = int(next(g for g in year_m.groups() if g is not None))
            date_from, date_to = f"{y}-01-01", f"{y + 1}-01-01"

    # Named institution — $in on entity_name (more specific than entity_type)
    canonical = extract_entities(question)
    if canonical:
        if len(canonical) == 1:
            chroma_conds.append({"entity_name": {"$eq": canonical[0]}})
        else:
            chroma_conds.append({"entity_name": {"$in": canonical}})
    else:
        # Entity type — only when no named institution detected
        if _NBFC_RE.search(question):
            chroma_conds.append({"entity_type": {"$in": NBFC_ENTITY_TYPES}})
        elif _COOP_BANK_RE.search(question):
            chroma_conds.append({"entity_type": {"$in": COOP_ENTITY_TYPES}})

    if not chroma_conds:
        chroma_where = None
    elif len(chroma_conds) == 1:
        chroma_where = chroma_conds[0]
    else:
        chroma_where = {"$and": chroma_conds}

    return chroma_where, date_from, date_to


# ── Date post-filter ──────────────────────────────────────────────────────────

def _apply_date_filter(
    chunks: list[VectorResult],
    date_from: str | None,
    date_to:   str | None,
) -> list[VectorResult]:
    """Post-filter chunks by date using Python string comparison (ISO 8601 sorts correctly)."""
    if not date_from and not date_to:
        return chunks
    filtered = [
        c for c in chunks
        if (date_from is None or c["date"] >= date_from)
        and (date_to   is None or c["date"] <  date_to)
    ]
    log.info(
        "advanced_vector | date post-filter (%s..%s): %d → %d chunks",
        date_from, date_to, len(chunks), len(filtered),
    )
    return filtered


# ── HyDE generation ───────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    reraise=True,
)
def _generate_hypothesis(question: str, cfg: dict) -> str:
    """Generate a hypothetical RBI press release passage for embedding (HyDE)."""
    msg = _get_client().messages.create(
        model       = cfg["models"]["system_model"],
        max_tokens  = 512,
        temperature = cfg["llm"]["hyde_temperature"],
        messages    = [{"role": "user", "content": _HYDE_TEMPLATE.format(question=question)}],
    )
    return msg.content[0].text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_advanced(
    question: str,
    cfg: dict | None = None,
    *,
    filter_q: str | None = None,
) -> list[VectorResult]:
    """
    Full advanced vector retrieval: HyDE + metadata filter + cosine + date filter + rerank.

    Args:
        question : Resolved query — used for HyDE hypothesis generation and cosine search.
        cfg      : Config dict. Loaded from get_config() if None.
        filter_q : If provided, extract metadata filters from this string instead of
                   *question*. Pass the pre-resolution template question in dependent hops
                   to prevent entity contamination from {q0.entity_name} substitution.

    Returns:
        Reranked list of VectorResult dicts (up to cfg["agentic_rag"]["top_k_vector_final"]).
        Empty list if no chunks retrieved or all filtered out.

    Graceful degradation:
        - HyDE failure → falls back to raw question embedding.
        - Reranker failure → falls back to cosine-ranked top_k_final.
    """
    if cfg is None:
        cfg = get_config()

    filter_question = filter_q if filter_q is not None else question

    # Step 1 — filter extraction (from template question to avoid contamination)
    chroma_where, date_from, date_to = _extract_filters(filter_question)
    log.debug(
        "advanced_vector | filters: where=%s date=%s..%s | q=%s",
        chroma_where, date_from, date_to, question[:60],
    )

    # Step 2 — HyDE (using resolved question for accurate hypothesis)
    try:
        hypothesis = _generate_hypothesis(question, cfg)
        log.info("advanced_vector | HyDE: %s", hypothesis[:100].replace("\n", " "))
    except Exception as exc:
        log.warning("advanced_vector | HyDE failed — raw query fallback: %s", exc)
        hypothesis = question

    # Step 3 — wide cosine retrieval with ChromaDB pre-filter
    top_k = cfg["advanced_rag"]["top_k_initial"]
    chunks = vector_retrieve(hypothesis, top_k=top_k, cfg=cfg, where=chroma_where)
    if not chunks:
        log.info("advanced_vector | 0 chunks from ChromaDB | %s", question[:60])
        return []

    # Step 4 — Python date post-filter
    chunks = _apply_date_filter(chunks, date_from, date_to)
    if not chunks:
        log.info("advanced_vector | 0 chunks after date filter | %s", question[:60])
        return []

    # Step 5 — cross-encoder reranking against original (resolved) question
    top_k_final = cfg["agentic_rag"]["top_k_vector_final"]
    try:
        reranked = rerank(question, chunks, cfg, top_k=top_k_final)
    except Exception as exc:
        log.error("advanced_vector | rerank failed, cosine fallback: %s", exc)
        reranked = chunks[:top_k_final]

    log.info(
        "advanced_vector | %d → %d chunks | entity_filter=%s | %s",
        len(chunks), len(reranked), chroma_where is not None, question[:60],
    )
    return reranked
