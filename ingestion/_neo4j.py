"""
ingestion/_neo4j.py

Neo4j driver construction and safe Cypher execution shared across all modules
that talk to Neo4j (graph_writer, graph_retriever, graph_metrics …).

Public API:
    build_driver(cfg)                            -> neo4j.Driver
    execute_cypher_safe(query, driver, params)   -> list[dict]
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, ServiceUnavailable
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger(__name__)


# ── Driver factory ────────────────────────────────────────────────────────────

def build_driver(cfg: dict):
    """
    Build Neo4j driver with resilience settings from config.yaml.
    Reads NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD from .env.
    Connection params: max_connection_lifetime + keep_alive per CLAUDE.md §6.
    """
    load_dotenv()
    ncfg = cfg["neo4j"]
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        max_connection_lifetime=ncfg["max_connection_lifetime"],
        connection_timeout=ncfg["connection_timeout"],
        keep_alive=True,
    )


# ── Error logging ─────────────────────────────────────────────────────────────

def _log_cypher_error(query: str, exc: Exception) -> None:
    entry = json.dumps({
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "module":     "neo4j",
        "error_type": type(exc).__name__,
        "query":      query[:500],
        "traceback":  traceback.format_exc(),
    })
    with (_LOG_DIR / "cypher_errors.log").open("a", encoding="utf-8") as fh:
        fh.write(entry + "\n")
    log.error("Cypher error: %s | %.120s", exc, query)


# ── Safe execution ────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(ServiceUnavailable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=4),
    reraise=True,
)
def execute_cypher_safe(query: str, driver, params: dict | None = None) -> list[dict]:
    """
    Execute Cypher safely.
      CypherSyntaxError  → logged to cypher_errors.log, re-raised
      ServiceUnavailable → retried ×3 with exponential backoff

    ALL Neo4j calls in this codebase must go through this function.
    Never call driver.execute_query() or session.run() directly.
    """
    try:
        with driver.session() as session:
            result = session.run(query, parameters=params or {})
            return [dict(record) for record in result]
    except CypherSyntaxError as exc:
        _log_cypher_error(query, exc)
        raise
