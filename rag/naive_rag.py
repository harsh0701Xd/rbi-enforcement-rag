"""
rag/naive_rag.py

Stage 1 -- Naive RAG: embed query -> ChromaDB top-k -> Claude synthesis.

No reranking, no graph traversal, no query decomposition.
Serves as the baseline in the three-stage RAGAS comparative study.

Public API:
    NaiveResult              -- TypedDict returned by answer()
    answer(question, cfg)    -- full Stage 1 pipeline, never raises
"""

from __future__ import annotations

import logging
import time
from typing import TypedDict

from dotenv import load_dotenv

from config import get_config
from prompts.synthesis import SYSTEM as _SYSTEM
from rag._logger import InferenceLogger
from rag._synthesis import synthesise as _synthesise
from retrieval.vector_retriever import VectorResult, retrieve as vector_retrieve

load_dotenv()

log = logging.getLogger(__name__)


# -- Return type --------------------------------------------------------------

class NaiveResult(TypedDict):
    question: str        # original question (passed through for RAGAS)
    answer:   str        # synthesised answer
    contexts: list[str]  # retrieved chunk texts (RAGAS context_precision / recall)
    stage:    str        # always "naive" -- for MLflow run tagging
    model:    str        # model name from config (logged to MLflow)


# -- Main pipeline ------------------------------------------------------------

def answer(question: str, cfg: dict | None = None) -> NaiveResult:
    """
    Stage 1 Naive RAG pipeline.

      1. Retrieve top-k chunks from ChromaDB via nomic-embed-text similarity
      2. Concatenate chunks into a numbered context window
      3. Synthesise answer with Claude (temperature=0)

    Never raises -- any retrieval or synthesis failure returns a NaiveResult
    with an informative answer string so callers (eval pipeline, API) stay stable.

    Args:
        question: Natural-language question.
        cfg:      Config dict from get_config(). Loaded automatically if None.

    Returns:
        NaiveResult with answer, contexts list, stage="naive", and model name.
    """
    if cfg is None:
        cfg = get_config()

    top_k: int = cfg["naive_rag"]["top_k"]
    model: str = cfg["models"]["system_model"]
    logger = InferenceLogger("naive", question)

    # Step 1: vector retrieval
    _t = time.perf_counter()
    chunks: list[VectorResult] = vector_retrieve(question, top_k=top_k, cfg=cfg)
    logger.step(
        "embed_retrieve",
        latency_s = time.perf_counter() - _t,
        n_chunks  = len(chunks),
        top_score = float(chunks[0]["score"]) if chunks else 0.0,
    )

    if not chunks:
        log.warning("No chunks retrieved | question=%s", question)
        ans = "I don't have enough information to answer this."
        logger.finish(ans, n_contexts=0)
        return NaiveResult(
            question=question, answer=ans,
            contexts=[], stage="naive", model=model,
        )

    log.info("Retrieved %d chunks | top_score=%.3f", len(chunks), chunks[0]["score"])

    # Step 2: build context window
    # Expose entity_name, date, pr_number so the model can produce inline citations.
    context = "\n\n".join(
        f"[{i + 1}] Entity: {chunk['entity_name']} | Date: {chunk['date']} | PR: {chunk['pr_number']}\n{chunk['text']}"
        for i, chunk in enumerate(chunks)
    )

    # Step 3: synthesise
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
    logger.finish(ans, n_contexts=len(chunks))

    return NaiveResult(
        question=question,
        answer=ans,
        contexts=[c["text"] for c in chunks],
        stage="naive",
        model=model,
    )


# -- Smoke test ---------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    _QUESTIONS = [
        "What penalty did RBI impose on Axis Bank and for what reason?",
        "What are KYC directions and why do banks get penalised for violating them?",
        "Which cooperative banks received RBI penalties for non-compliance?",
    ]

    for q in _QUESTIONS:
        print(f"\nQ: {q}")
        result = answer(q)
        print(f"A: {result['answer']}")
        print(f"   contexts={len(result['contexts'])}  stage={result['stage']}")
