"""
retrieval/vector_retriever.py

Embeds a natural-language query via Ollama and retrieves the most similar
chunks from the ChromaDB collection (cfg["chroma"]["collection_name"]).

Used by all three RAG stages:
  - Stage 1 (Naive RAG):    retrieve(query, top_k=cfg["naive_rag"]["top_k"])
  - Stage 2 (Advanced RAG): retrieve(query, top_k=cfg["advanced_rag"]["top_k_initial"], where=filters)
  - Stage 3 (Agentic):      retrieve(query, top_k=cfg["agentic_rag"]["graph_fallback_threshold"])

Public API:
    VectorResult            -- TypedDict for a single retrieved chunk
    retrieve(query, top_k, cfg, where)  -- main retrieval function
"""

from __future__ import annotations

import logging
import os
from typing import TypedDict

import chromadb
import httpx
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_config
from ingestion._text_processor import OllamaUnavailableError

log = logging.getLogger(__name__)

load_dotenv()


class VectorResult(TypedDict):
    text:         str
    prid:         int
    entity_name:  str
    entity_type:  str    # e.g. "Bank-Private", "NBFC", "Payment System Operator"
    date:         str    # YYYY-MM-DD
    action_type:  str    # "monetary_penalty" | "cor_cancellation" | "other"
    source_url:   str
    pr_number:    str
    chunk_index:  int
    total_chunks: int
    score:        float  # cosine similarity in [0, 1]; higher = more similar


# Module-level ChromaDB singleton -- opened once per process to avoid
# re-loading the HNSW index on every query.

_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        cfg         = get_config()
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
        col_name    = cfg["chroma"]["collection_name"]
        client      = chromadb.PersistentClient(path=persist_dir)
        _collection = client.get_collection(col_name)
        log.debug("ChromaDB collection '%s' opened (%s)", col_name, persist_dir)
    return _collection


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=4),
    reraise=True,
)
def _embed_query(text: str, cfg: dict) -> list[float]:
    """
    Embed a single query string via Ollama /api/embed.
    Uses the same model as ingestion so query and document spaces match.
    Raises OllamaUnavailableError if Ollama is not running.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model    = cfg["embedding"]["model"]
    try:
        resp = httpx.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": [text]},
            timeout=cfg["embedding"]["ollama"]["request_timeout"],
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            f"Ollama unreachable at {base_url} -- run `ollama serve` first."
        ) from exc


def retrieve(
    query: str,
    top_k: int,
    cfg: dict | None = None,
    where: dict | None = None,
) -> list[VectorResult]:
    """
    Embed query and return the top_k most similar chunks from ChromaDB.

    Args:
        query:  Natural-language question or sub-question.
        top_k:  Number of chunks to return. Pass cfg value for the active stage:
                  Stage 1 -> cfg["naive_rag"]["top_k"]            (5)
                  Stage 2 -> cfg["advanced_rag"]["top_k_initial"]  (10)
        cfg:    Config dict from get_config(). Loaded automatically if None.
        where:  Optional ChromaDB metadata pre-filter applied before similarity
                search. Constructed by advanced_rag._extract_filters(). Examples:
                  {"date": {"$gte": "2023-01-01"}}
                  {"$and": [{"date": {"$gte": "2023-01-01"}},
                             {"entity_type": {"$eq": "NBFC"}}]}
                None means no filtering (Stage 1 behaviour).

    Returns:
        List of VectorResult dicts sorted by descending cosine similarity.
        Empty list if ChromaDB has no matching chunks (not an error).
    """
    if cfg is None:
        cfg = get_config()

    embedding  = _embed_query(query, cfg)
    collection = _get_collection()

    query_kwargs: dict = dict(
        query_embeddings=[embedding],
        n_results=top_k,  # ChromaDB returns min(n_results, actual_count) automatically
        include=["documents", "metadatas", "distances"],
    )
    if where is not None:
        query_kwargs["where"] = where

    raw = collection.query(**query_kwargs)

    results: list[VectorResult] = []
    docs      = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]  # cosine distance in [0, 2]; 0 = identical

    for doc, meta, dist in zip(docs, metadatas, distances):
        results.append(VectorResult(
            text         = doc,
            prid         = int(meta.get("prid", 0)),
            entity_name  = meta.get("entity_name", ""),
            entity_type  = meta.get("entity_type", ""),
            date         = meta.get("date", ""),
            action_type  = meta.get("action_type", ""),
            source_url   = meta.get("source_url", ""),
            pr_number    = meta.get("pr_number", ""),
            chunk_index  = int(meta.get("chunk_index", 0)),
            total_chunks = int(meta.get("total_chunks", 0)),
            score        = round(1.0 - dist, 4),  # convert distance to similarity
        ))

    log.debug("retrieve('%s...', top_k=%d) -> %d results", query[:40], top_k, len(results))
    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    _cfg = get_config()
    test_queries = [
        "Axis Bank KYC penalty",
        "Which NBFC had its CoR cancelled for FEMA violations?",
        "RBI penalty on cooperative bank for non-compliance with directions",
    ]
    for q in test_queries:
        hits = retrieve(q, top_k=3, cfg=_cfg)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['entity_name']} | prid={h['prid']} | {h['date']}")
            print(f"         {h['text'][:120]}...")
