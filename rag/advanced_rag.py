"""
rag/advanced_rag.py

Stage 2 -- Advanced RAG: metadata filtering + HyDE + cross-encoder reranking.

Improvements over Stage 1 (naive_rag.py):
  1. Metadata filtering  -- deterministic regex extracts filter intent from the
     query. Two filter types are separated:
       chroma_where : entity_type / action_type passed as ChromaDB pre-filters
                      (ChromaDB supports $eq on string fields only)
       date_from/to : applied as Python post-filters after retrieval
                      (ChromaDB requires numeric fields for $gte/$lt; our date
                      field is ISO string "YYYY-MM-DD" which sorts correctly
                      with lexicographic comparison)
  2. HyDE               -- embeds a Claude-generated hypothetical RBI press
     release instead of the raw query, closing the vocabulary gap.
  3. Wide retrieval + reranking -- top_k_initial=10 cosine on hypothesis, then
     cross-encoder rerank to top_k_final=3.

Public API:
    AdvancedResult             -- TypedDict returned by answer()
    answer(question, cfg)      -- full Stage 2 pipeline, never raises
"""

from __future__ import annotations

import logging
import re
import time
from typing import TypedDict

import anthropic
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_config
from rag._entities import COOP_ENTITY_TYPES, NBFC_ENTITY_TYPES, extract_entities
from rag._logger import InferenceLogger
from rag._reranker import rerank
from retrieval.vector_retriever import VectorResult, retrieve as vector_retrieve

load_dotenv()

log = logging.getLogger(__name__)


# ── Return type ───────────────────────────────────────────────────────────────

class AdvancedResult(TypedDict):
    question:   str
    answer:     str
    contexts:   list[str]
    hypothesis: str
    filters:    dict        # always present: {chroma_where, date_from, date_to}
    stage:      str
    model:      str


# ── Prompts ───────────────────────────────────────────────────────────────────
from prompts.synthesis import SYSTEM as _SYSTEM
from rag._synthesis import synthesise as _synthesise, get_client as _get_client
from prompts.hyde import TEMPLATE as _HYDE_TEMPLATE


# ── Deterministic metadata filter extraction ──────────────────────────────────
# ChromaDB constraint: $gte/$lt operators only work on numeric fields.
# Our "date" field is stored as ISO string "YYYY-MM-DD".
# Solution:
#   chroma_where -> entity_type only (string $in/$eq, supported by ChromaDB)
#   date_from/to -> returned separately for Python post-filtering
#                   (ISO strings sort correctly with lexicographic comparison)
#
# NOTE: action_type "cor_cancellation" does NOT exist in the vector store.
# All CoR cancellations are bulk press releases (is_bulk_cancellation=True)
# which were excluded from ingestion. CoR queries must go to Stage 3 (graph).
# Observed action_types: monetary_penalty (1118), other (182).
#
# NOTE: NBFC entity_type is subcategorized in the corpus -- actual values are
# NBFC-ICC, NBFC-HFC, NBFC-P2P, NBFC-MFI. Use $in to match all subtypes.
# Observed entity_types: Bank-Cooperative (671), Bank-UCB (215),
# Bank-Private (105), NBFC-ICC (83), Bank-DCCB (59), Bank-Public (52),
# NBFC-HFC (27), Bank-SCB (22), PPI (18), Bank-RRB (12), NBFC-P2P (10),
# CIC (6), Bank-Payment (6), Bank-SFB (6), NBFC-MFI (4), PA (2), ARC (1).

_YEAR_RANGE_RE  = re.compile(r'\b(20\d{2})\s+to\s+(20\d{2})\b', re.IGNORECASE)
_SINGLE_YEAR_RE = re.compile(r'\bin\s+(20\d{2})\b|\bduring\s+(20\d{2})\b|\b(20\d{2})\b')
_NBFC_RE        = re.compile(r'\bNBFCs?\b|\bnon.?banking financial\b', re.IGNORECASE)
_COOP_BANK_RE   = re.compile(r'\bco.?operative\s+bank\b|\burban\s+co.?op\b', re.IGNORECASE)
# NBFC_ENTITY_TYPES and COOP_ENTITY_TYPES imported from rag._entities


def _extract_filters(question: str) -> tuple[dict | None, str | None, str | None]:
    """
    Extract metadata filters from a natural-language question.

    Returns:
        chroma_where : ChromaDB where clause, or None if no filters found.
        date_from    : "YYYY-MM-DD" inclusive lower bound, or None.
        date_to      : "YYYY-MM-DD" exclusive upper bound, or None.

    Filter priority:
        1. Named institution detected (via rag._entities.extract_entities)
           -> entity_name $in [canonical variants]
           -> entity_type filter is SKIPPED (named entity is more specific)
        2. No named institution, but NBFC/co-op keyword detected
           -> entity_type $in [NBFC subtypes] or [coop subtypes]
        3. Date range/year detected (always applied as Python post-filter,
           independent of entity filters)

    NOT supported (data gap -- use Stage 3 graph):
        CoR cancellations -> no cor_cancellation records in vector store
    """
    chroma_conds: list[dict] = []
    date_from: str | None = None
    date_to:   str | None = None

    # Date range -- post-filtered in Python, NOT passed to ChromaDB
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

    # Named institution -- $in on entity_name, ChromaDB-compatible.
    # When a specific institution is named, skip entity_type filter:
    # "Axis Bank" questions are more specifically answered by entity_name
    # than by entity_type ("Bank-Private"), and stacking both with $and
    # would over-restrict results.
    canonical = extract_entities(question)
    if canonical:
        if len(canonical) == 1:
            chroma_conds.append({"entity_name": {"$eq": canonical[0]}})
        else:
            chroma_conds.append({"entity_name": {"$in": canonical}})
    else:
        # Entity type -- only when no named institution detected
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


def _apply_date_filter(
    chunks: list[VectorResult],
    date_from: str | None,
    date_to: str | None,
) -> list[VectorResult]:
    """
    Post-filter retrieved chunks by date using Python string comparison.
    ISO 8601 date strings ("YYYY-MM-DD") are lexicographically ordered, so
    string comparison gives the correct chronological result.
    """
    if not date_from and not date_to:
        return chunks
    filtered = [
        c for c in chunks
        if (date_from is None or c["date"] >= date_from)
        and (date_to   is None or c["date"] <  date_to)
    ]
    log.info("Date post-filter (%s .. %s): %d -> %d chunks",
             date_from, date_to, len(chunks), len(filtered))
    return filtered



# ── HyDE generation ───────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    reraise=True,
)
def _generate_hypothesis(question: str, cfg: dict) -> str:
    msg = _get_client().messages.create(
        model=cfg["models"]["system_model"],
        max_tokens=512,
        temperature=cfg["llm"]["hyde_temperature"],
        messages=[{"role": "user", "content": _HYDE_TEMPLATE.format(question=question)}],
    )
    return msg.content[0].text.strip()



# ── Main pipeline ─────────────────────────────────────────────────────────────

def answer(question: str, cfg: dict | None = None) -> AdvancedResult:
    """
    Stage 2 Advanced RAG pipeline.

      1. Extract filters: chroma_where (entity_type/action_type) + date range
      2. Generate HyDE: hypothetical RBI press release at temperature=0.3
      3. Wide cosine retrieval (top_k_initial=10) with chroma_where pre-filter
      4. Python post-filter by date range (ISO string comparison)
      5. Cross-encoder reranking (top_k_final=3) against original question
      6. Synthesise answer with Claude, citing sources inline

    Graceful degradation at every step -- never raises.
    """
    if cfg is None:
        cfg = get_config()

    model: str = cfg["models"]["system_model"]
    logger = InferenceLogger("advanced", question)

    # Step 1: filter extraction (deterministic — not timed)
    chroma_where, date_from, date_to = _extract_filters(question)
    filters = {"chroma_where": chroma_where, "date_from": date_from, "date_to": date_to}
    log.info("Filters: chroma_where=%s  date=%s..%s", chroma_where, date_from, date_to)

    # Step 2: HyDE
    hyde_fallback = False
    _t = time.perf_counter()
    try:
        hypothesis = _generate_hypothesis(question, cfg)
        log.info("HyDE: %s", hypothesis[:120].replace("\n", " "))
    except Exception as exc:
        log.error("HyDE failed -- falling back to raw question: %s", exc)
        hypothesis = question
        hyde_fallback = True
    logger.step(
        "hyde",
        latency_s    = time.perf_counter() - _t,
        hyde_fallback = hyde_fallback,
    )

    # Step 3: wide cosine retrieval with ChromaDB pre-filter
    top_k_initial: int = cfg["advanced_rag"]["top_k_initial"]
    _t = time.perf_counter()
    chunks: list[VectorResult] = vector_retrieve(
        hypothesis, top_k=top_k_initial, cfg=cfg, where=chroma_where,
    )
    logger.step(
        "embed_retrieve",
        latency_s       = time.perf_counter() - _t,
        n_chunks        = len(chunks),
        top_score       = float(chunks[0]["score"]) if chunks else 0.0,
        filter_applied  = chroma_where is not None,
    )

    if not chunks:
        log.warning("No chunks retrieved after ChromaDB filter | question=%s", question)
        ans = "I don't have enough information to answer this."
        logger.finish(ans, n_contexts=0)
        return AdvancedResult(
            question=question, answer=ans,
            contexts=[], hypothesis=hypothesis, filters=filters,
            stage="advanced", model=model,
        )

    log.info("Wide retrieval: %d chunks | top_score=%.3f | entities: %s",
             len(chunks), chunks[0]["score"], [c["entity_name"] for c in chunks[:3]])

    # Step 4: Python date post-filter
    n_before_date = len(chunks)
    _t = time.perf_counter()
    chunks = _apply_date_filter(chunks, date_from, date_to)
    logger.step(
        "date_filter",
        latency_s         = time.perf_counter() - _t,
        date_filter_active = date_from is not None or date_to is not None,
        n_before          = n_before_date,
        n_after           = len(chunks),
    )

    if not chunks:
        log.warning("No chunks after date post-filter (%s..%s)", date_from, date_to)
        ans = "I don't have enough information to answer this."
        logger.finish(ans, n_contexts=0)
        return AdvancedResult(
            question=question, answer=ans,
            contexts=[], hypothesis=hypothesis, filters=filters,
            stage="advanced", model=model,
        )

    # Step 5: cross-encoder reranking against original question
    top_reranker_score: float = 0.0
    _t = time.perf_counter()
    try:
        reranked, reranker_scores = rerank(question, chunks, cfg, return_scores=True)
        top_reranker_score = reranker_scores[0] if reranker_scores else 0.0
    except Exception as exc:
        log.error("Reranking failed -- using cosine top-%d: %s",
                  cfg["advanced_rag"]["top_k_final"], exc)
        reranked = chunks[:cfg["advanced_rag"]["top_k_final"]]
    logger.step(
        "rerank",
        latency_s         = time.perf_counter() - _t,
        n_chunks_final    = len(reranked),
        top_reranker_score = top_reranker_score,
    )

    log.info("After reranking: %d chunks | entities: %s",
             len(reranked), [c["entity_name"] for c in reranked])

    # Step 6: synthesise
    context = "\n\n".join(
        f"[{i + 1}] Entity: {c['entity_name']} | Date: {c['date']} | PR: {c['pr_number']}\n{c['text']}"
        for i, c in enumerate(reranked)
    )
    ans: str
    usage: dict | None = None
    _t = time.perf_counter()
    try:
        ans, usage = _synthesise(context, question, cfg, _SYSTEM, return_usage=True)
    except Exception as exc:
        log.error("Synthesis failed: %s", exc)
        ans = "An error occurred while generating the answer. Please try again."
    logger.step(
        "synthesis",
        latency_s     = time.perf_counter() - _t,
        input_tokens  = (usage or {}).get("input_tokens",  0),
        output_tokens = (usage or {}).get("output_tokens", 0),
    )

    log.info("Answer generated | chars=%d", len(ans))
    logger.finish(ans, n_contexts=len(reranked))

    return AdvancedResult(
        question=question, answer=ans,
        contexts=[c["text"] for c in reranked],
        hypothesis=hypothesis, filters=filters,
        stage="advanced", model=model,
    )


# ── Smoke test ───