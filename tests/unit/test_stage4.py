"""
tests/unit/test_stage4.py

Unit tests for Stage 4 Agentic RAG upgrades.
No Ollama, no ChromaDB, no Neo4j, no Anthropic API -- all external I/O mocked.

Covers three upgrade areas:
  1. _graph_rows_to_context_strings()  -- new helper in agentic_rag.py
  2. vector_node upgrades              -- entity filter, wider top_k, reranking
  3. AgenticResult.contexts            -- vector + graph row strings merged

Run from rbi_intel/:
    pytest tests/unit/test_stage4.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag.agentic_rag import _graph_rows_to_context_strings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_chunk(**overrides) -> dict:
    base = {
        "text":         "RBI imposed a monetary penalty.",
        "prid":         1,
        "entity_name":  "Test Bank Limited",
        "entity_type":  "Bank-Private",
        "date":         "2023-06-01",
        "action_type":  "monetary_penalty",
        "source_url":   "https://rbi.org.in/test",
        "pr_number":    "2023-2024/123",
        "chunk_index":  0,
        "total_chunks": 1,
        "score":        0.9,
    }
    base.update(overrides)
    return base


def _fake_graph_result(rows: list[dict], success: bool = True) -> dict:
    return {
        "cypher":  "MATCH (i:Institution)-[:RECEIVED]->(p) RETURN i.name, p.penalty_lakh",
        "rows":    rows,
        "success": success,
    }


_BASE_CFG = {
    "models":       {"system_model": "claude-haiku-4-5-20251001",
                     "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2"},
    "llm":          {"chat_temperature": 0, "max_tokens_chat": 512},
    "naive_rag":    {"top_k": 5},
    "advanced_rag": {"top_k_initial": 10, "top_k_final": 3},
    "agentic_rag":  {"max_sub_questions": 4, "max_iterations": 5,
                     "cypher_limit": 20, "graph_fallback_threshold": 0,
                     "top_k_vector": 10, "top_k_vector_final": 5},
    "embedding":    {"model": "nomic-embed-text",
                     "ollama": {"request_timeout": 30}},
    "chroma":       {"collection_name": "rbi_enforcement"},
}


# ── 1. TestGraphRowsToContextStrings ─────────────────────────────────────────

class TestGraphRowsToContextStrings:
    """Tests for agentic_rag._graph_rows_to_context_strings()."""

    def test_basic_conversion(self):
        results = [_fake_graph_result([
            {"i.name": "Axis Bank Limited", "p.penalty_lakh": 1.5, "p.date": "2023-03-15"},
        ])]
        strings = _graph_rows_to_context_strings(results)
        assert len(strings) == 1
        assert "Axis Bank Limited" in strings[0]
        assert "1.5" in strings[0]

    def test_alias_prefix_stripped(self):
        # "i.name", "p.penalty_lakh", "v.text" → "name", "penalty_lakh", "text"
        results = [_fake_graph_result([
            {"i.name": "SBI", "p.penalty_lakh": 2.0, "v.text": "KYC violation"},
        ])]
        s = _graph_rows_to_context_strings(results)[0]
        assert "i." not in s and "p." not in s and "v." not in s
        assert "name:" in s

    def test_none_fields_excluded(self):
        results = [_fake_graph_result([
            {"i.name": "Axis Bank Limited", "p.penalty_lakh": None, "p.date": "2023-03-15"},
        ])]
        s = _graph_rows_to_context_strings(results)[0]
        assert "None" not in s
        assert "penalty_lakh" not in s

    def test_empty_string_fields_excluded(self):
        results = [_fake_graph_result([
            {"i.name": "Axis Bank Limited", "p.pr_number": "  ", "p.date": "2023-03-15"},
        ])]
        s = _graph_rows_to_context_strings(results)[0]
        assert "pr_number" not in s

    def test_duplicate_rows_collapsed(self):
        row = {"i.name": "Axis Bank Limited", "p.penalty_lakh": 1.5}
        results = [_fake_graph_result([row, row, row])]  # same row 3 times
        strings = _graph_rows_to_context_strings(results)
        assert len(strings) == 1

    def test_duplicates_across_graph_results_collapsed(self):
        row = {"i.name": "HDFC Bank Limited", "p.date": "2023-06-01"}
        results = [
            _fake_graph_result([row]),
            _fake_graph_result([row]),   # same row from different sub-question
        ]
        strings = _graph_rows_to_context_strings(results)
        assert len(strings) == 1

    def test_failed_results_skipped(self):
        results = [
            _fake_graph_result([{"i.name": "Ignored"}], success=False),
            _fake_graph_result([{"i.name": "Included", "p.date": "2023-01-01"}]),
        ]
        strings = _graph_rows_to_context_strings(results)
        assert len(strings) == 1
        assert "Included" in strings[0]
        assert "Ignored" not in strings[0]

    def test_empty_rows_skipped(self):
        results = [_fake_graph_result([], success=True)]
        assert _graph_rows_to_context_strings(results) == []

    def test_all_none_row_produces_no_string(self):
        results = [_fake_graph_result([{"i.name": None, "p.penalty_lakh": None}])]
        assert _graph_rows_to_context_strings(results) == []

    def test_empty_input(self):
        assert _graph_rows_to_context_strings([]) == []

    def test_multiple_rows_multiple_results(self):
        results = [
            _fake_graph_result([
                {"i.name": "Axis Bank Limited", "p.penalty_lakh": 1.5},
                {"i.name": "HDFC Bank Limited", "p.penalty_lakh": 2.0},
            ]),
            _fake_graph_result([
                {"i.name": "Kotak Mahindra Bank Limited", "p.date": "2023-09-01"},
            ]),
        ]
        strings = _graph_rows_to_context_strings(results)
        assert len(strings) == 3
        entity_names = " ".join(strings)
        assert "Axis Bank Limited" in entity_names
        assert "HDFC Bank Limited" in entity_names
        assert "Kotak Mahindra Bank Limited" in entity_names

    def test_pipe_separator_between_fields(self):
        results = [_fake_graph_result([
            {"i.name": "Test Bank", "p.penalty_lakh": 1.0, "p.date": "2023-01-01"},
        ])]
        s = _graph_rows_to_context_strings(results)[0]
        assert " | " in s


# ── 2. TestVectorNodeUpgrades ─────────────────────────────────────────────────

class TestVectorNodeUpgrades:
    """Tests for the Stage 4 vector_node: entity filter + top_k + reranking."""

    _STATE = {
        "question":      "What penalty did RBI impose on Axis Bank?",
        "sub_questions": ["What penalty did RBI impose on Axis Bank?"],
        "routes":        ["both"],
        "vector_chunks": [],
        "graph_results": [],
        "context":       "",
        "answer":        "",
        "iterations":    0,
    }

    def _run_vector_node(self, sub_questions, routes, mock_retrieve, mock_rerank):
        from rag.nodes.retrieval import vector_node
        state = {**self._STATE, "sub_questions": sub_questions, "routes": routes}
        with patch("rag.nodes.retrieval.vector_retrieve", mock_retrieve), \
             patch("rag.nodes.retrieval.rerank", mock_rerank), \
             patch("rag.nodes.retrieval.get_config", return_value=_BASE_CFG):
            return vector_node(state)

    def test_uses_agentic_top_k_not_naive(self):
        """vector_node must use agentic_rag.top_k_vector (10), not naive_rag.top_k (5)."""
        mock_retrieve = MagicMock(return_value=[_fake_chunk()])
        mock_rerank   = MagicMock(return_value=[_fake_chunk()])
        self._run_vector_node(["Axis Bank penalty"], ["both"], mock_retrieve, mock_rerank)
        called_top_k = mock_retrieve.call_args[1]["top_k"]
        assert called_top_k == 10, f"expected top_k=10, got {called_top_k}"

    def test_entity_filter_applied_for_named_institution(self):
        """Named entity in sub-question -> where filter passed to vector_retrieve."""
        mock_retrieve = MagicMock(return_value=[_fake_chunk()])
        mock_rerank   = MagicMock(return_value=[_fake_chunk()])
        self._run_vector_node(["What penalty did RBI impose on Axis Bank?"],
                              ["both"], mock_retrieve, mock_rerank)
        where = mock_retrieve.call_args[1].get("where")
        assert where is not None, "entity filter should be applied for Axis Bank"
        assert "entity_name" in str(where)
        assert "Axis Bank Limited" in str(where)

    def test_no_entity_filter_for_generic_query(self):
        """Generic query -> where=None (full corpus scan)."""
        mock_retrieve = MagicMock(return_value=[_fake_chunk()])
        mock_rerank   = MagicMock(return_value=[_fake_chunk()])
        self._run_vector_node(["Which banks violated KYC?"],
                              ["vector"], mock_retrieve, mock_rerank)
        where = mock_retrieve.call_args[1].get("where")
        assert where is None

    def test_rerank_called_with_agentic_top_k_final(self):
        """Reranker must use agentic_rag.top_k_vector_final (5), not advanced_rag.top_k_final (3)."""
        fake = [_fake_chunk(entity_name=f"Bank {i}") for i in range(10)]
        mock_retrieve = MagicMock(return_value=fake)
        mock_rerank   = MagicMock(return_value=fake[:5])
        self._run_vector_node(["Axis Bank penalty"], ["both"], mock_retrieve, mock_rerank)
        called_top_k = mock_rerank.call_args[1].get("top_k")
        assert called_top_k == 5, f"expected rerank top_k=5, got {called_top_k}"

    def test_chunk_text_includes_metadata_prefix(self):
        """Chunks returned by vector_node must include Entity/Date/PR metadata prefix."""
        chunk = _fake_chunk(entity_name="Axis Bank Limited",
                            date="2023-03-15", pr_number="2023-2024/100")
        mock_retrieve = MagicMock(return_value=[chunk])
        mock_rerank   = MagicMock(return_value=[chunk])
        result = self._run_vector_node(["Axis Bank penalty"], ["both"],
                                       mock_retrieve, mock_rerank)
        chunks = result["vector_chunks"]
        assert len(chunks) == 1
        assert "Entity: Axis Bank Limited" in chunks[0]
        assert "Date: 2023-03-15" in chunks[0]
        assert "PR: 2023-2024/100" in chunks[0]

    def test_graph_only_route_skipped(self):
        """Sub-questions routed 'graph' must not trigger vector retrieval."""
        mock_retrieve = MagicMock(return_value=[_fake_chunk()])
        mock_rerank   = MagicMock(return_value=[_fake_chunk()])
        self._run_vector_node(["How many banks were penalised?"],
                              ["graph"], mock_retrieve, mock_rerank)
        mock_retrieve.assert_not_called()

    def test_rerank_failure_falls_back_to_cosine_slice(self):
        """If reranker raises, vector_node must fall back to cosine top_k_final chunks."""
        # Each chunk needs unique text so the dedup set doesn't collapse them
        fake = [_fake_chunk(entity_name=f"Bank {i}", text=f"Penalty on Bank {i}.")
                for i in range(10)]
        mock_retrieve = MagicMock(return_value=fake)
        mock_rerank   = MagicMock(side_effect=RuntimeError("model not loaded"))
        result = self._run_vector_node(["Axis Bank penalty"], ["both"],
                                       mock_retrieve, mock_rerank)
        assert len(result["vector_chunks"]) == 5   # top_k_vector_final

    def test_cross_sub_question_dedup(self):
        """Same chunk text from two sub-questions must appear only once."""
        shared = _fake_chunk(text="RBI imposed penalty on Axis Bank.")
        mock_retrieve = MagicMock(return_value=[shared])
        mock_rerank   = MagicMock(return_value=[shared])
        result = self._run_vector_node(
            ["Axis Bank penalty Q1", "Axis Bank penalty Q2"],
            ["both", "both"],
            mock_retrieve, mock_rerank,
        )
        # Two sub-questions, same chunk -> deduplicated to 1
        texts = [c.split("\n", 1)[1] for c in result["vector_chunks"]]
        assert texts.count(shared["text"]) == 1


# ── 3. TestAgenticContextPassing ─────────────────────────────────────────────

class TestAgenticContextPassing:
    """
    Tests that AgenticResult.contexts = vector_chunks + graph_row_strings.
    The full pipeline is too heavy to mock end-to-end here; we test the merge
    logic via _graph_rows_to_context_strings + manual composition.
    """

    def test_contexts_include_graph_rows(self):
        vector_chunks = ["Entity: Axis Bank | Date: 2023-03-15 | PR: X\nSome text."]
        graph_results = [_fake_graph_result([
            {"i.name": "Axis Bank Limited", "p.penalty_lakh": 1.5, "p.date": "2023-03-15"},
        ])]
        graph_ctx = _graph_rows_to_context_strings(graph_results)
        all_contexts = vector_chunks + graph_ctx

        assert len(all_contexts) == 2
        assert any("Axis Bank Limited" in c for c in all_contexts)
        assert any("penalty_lakh" in c for c in all_contexts)

    def test_contexts_vector_only_when_no_graph(self):
        vector_chunks = ["chunk text"]
        graph_ctx = _graph_rows_to_context_strings([])
        all_contexts = vector_chunks + graph_ctx
        assert all_contexts == ["chunk text"]

    def test_contexts_graph_only_when_no_vector(self):
        graph_results = [_fake_graph_result([
            {"i.name": "HDFC Bank Limited", "p.penalty_lakh": 2.0},
        ])]
        graph_ctx = _graph_rows_to_context_strings(graph_results)
        all_contexts = [] + graph_ctx
        assert len(all_contexts) == 1
        assert "HDFC Bank Limited" in all_contexts[0]

    def test_entity_names_in_graph_context_strings(self):
        """Graph context strings must contain canonical entity names for RAGAS."""
        graph_results = [_fake_graph_result([
            {"i.name": "State Bank of India", "p.penalty_lakh": 5.0},
            {"i.name": "Punjab National Bank", "p.date": "2022-11-01"},
        ])]
        graph_ctx = _graph_rows_to_context_strings(graph_results)
        combined = " ".join(graph_ctx)
        assert "State Bank of India" in combined
        assert "Punjab National Bank" in combined
