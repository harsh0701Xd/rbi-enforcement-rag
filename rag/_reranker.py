"""
rag/_reranker.py

Cross-encoder reranking -- shared by advanced_rag.py and agentic_rag.py.

Loads cross-encoder/ms-marco-MiniLM-L6-v2 once per process (2-3s cold start)
and exposes a single rerank() function that scores (question, passage) pairs
and returns the top-k highest-scoring chunks.

Why cross-encoder over bi-encoder (cosine) for reranking:
  Bi-encoder: question and passage embedded independently -> fast but approximate.
  Cross-encoder: question + passage read jointly -> slower but precise relevance.
  Stage 2 uses bi-encoder for wide retrieval (speed) then cross-encoder to
  filter the top-10 down to top-3 (precision).
"""

from __future__ import annotations

import logging

from config import get_config
from retrieval.vector_retriever import VectorResult

log = logging.getLogger(__name__)

_reranker = None  # lazy-loaded on first call — avoids torch/transformers import at startup


def _get_reranker(cfg: dict):
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder  # deferred — keeps startup fast
        model = cfg["models"]["reranker_model"]
        log.info("Loading cross-encoder: %s", model)
        _reranker = CrossEncoder(model)
    return _reranker


def rerank(
    question: str,
    chunks: list[VectorResult],
    cfg: dict | None = None,
    *,
    top_k: int | None = None,
    return_scores: bool = False,
) -> "list[VectorResult] | tuple[list[VectorResult], list[float]]":
    """
    Score every (question, chunk_text) pair with the cross-encoder.
    Returns top_k chunks in descending relevance order.

    Args:
        top_k:         Number of chunks to keep. Defaults to
                       cfg["advanced_rag"]["top_k_final"] (3) when None.
                       Pass cfg["agentic_rag"]["top_k_vector_final"] (5) for
                       the Stage 4 agentic vector_node.
        return_scores: If True, return (chunks, scores) tuple instead of chunks
                       alone.  Scores are float logits from the cross-encoder,
                       useful for inference logging without changing the default
                       call signature.
    """
    if cfg is None:
        cfg = get_config()
    if top_k is None:
        top_k = cfg["advanced_rag"]["top_k_final"]
    pairs  = [(question, c["text"]) for c in chunks]
    scores = _get_reranker(cfg).predict(pairs)
    ranked     = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_pairs  = ranked[:top_k]
    top_chunks = [c for _, c in top_pairs]
    if return_scores:
        top_scores = [float(s) for s, _ in top_pairs]
        return top_chunks, top_scores
    return top_chunks
