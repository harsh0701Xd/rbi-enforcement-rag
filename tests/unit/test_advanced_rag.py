"""
tests/unit/test_advanced_rag.py

Unit tests for rag/advanced_rag.py.
No Ollama, no ChromaDB, no Anthropic API -- all external dependencies are mocked.

_extract_filters() returns (chroma_where, date_from, date_to):
  - chroma_where : ChromaDB $in filters (entity_type only) or None
  - date_from    : "YYYY-MM-DD" or None  (applied via Python post-filter)
  - date_to      : "YYYY-MM-DD" or None

Corpus constraints discovered via diagnostic:
  - action_type has only "monetary_penalty" and "other" -- no "cor_cancellation"
    (all CoR cancellations are bulk press releases, excluded from vector store)
  - entity_type is subcategorized: NBFC-ICC, NBFC-HFC, NBFC-P2P, NBFC-MFI
    (not plain "NBFC") -- filter uses $in across all subtypes

Run from rbi_intel/:
    pytest tests/unit/test_advanced_rag.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag._entities import COOP_ENTITY_TYPES as _COOP_ENTITY_TYPES
from rag._entities import NBFC_ENTITY_TYPES as _NBFC_ENTITY_TYPES
from rag.advanced_rag import (
    _apply_date_filter,
    _extract_filters,
    answer,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_chunk(**overrides) -> dict:
    base = {
        "text": "The Reserve Bank of India imposed a penalty.",
        "prid": 1, "entity_name": "Test Bank", "entity_type": "Bank-Private",
        "date": "2024-01-15", "action_type": "monetary_penalty",
        "source_url": "https://rbi.org.in/test", "pr_number": "2024-2025/1",
        "chunk_index": 0, "total_chunks": 1, "score": 0.9,
    }
    base.update(overrides)
    return base


# ── TestExtractFilters ────────────────────────────────────────────────────────

class TestExtractFilters:
    """Pure regex logic -- no mocking required."""

    # -- Single year (date_from/date_to only; chroma_where=None) --------------

    def test_single_year_in_phrase(self):
        where, df, dt = _extract_filters("Which banks were penalised in 2023?")
        assert where is None
        assert df == "2023-01-01" and dt == "2024-01-01"

    def test_single_year_during_phrase(self):
        _, df, dt = _extract_filters("Penalties issued during 2022")
        assert df == "2022-01-01" and dt == "2023-01-01"

    def test_single_year_bare(self):
        _, df, dt = _extract_filters("KYC violations 2024")
        assert df == "2024-01-01" and dt == "2025-01-01"

    # -- Year range -----------------------------------------------------------

    def test_year_range(self):
        _, df, dt = _extract_filters("Actions from 2022 to 2024")
        assert df == "2022-01-01" and dt == "2025-01-01"

    def test_year_range_reverse_order(self):
        _, df, dt = _extract_filters("Penalties 2024 to 2022")
        assert df == "2022-01-01" and dt == "2025-01-01"

    # -- NBFC entity_type filter ($in across all subtypes) --------------------

    def test_nbfc_singular_uses_dollar_in(self):
        where, df, dt = _extract_filters("Penalty imposed on an NBFC")
        assert where == {"entity_type": {"$in": _NBFC_ENTITY_TYPES}}
        assert df is None and dt is None

    def test_nbfc_plural_uses_dollar_in(self):
        where, _, __ = _extract_filters("Which NBFCs were penalised?")
        assert where == {"entity_type": {"$in": _NBFC_ENTITY_TYPES}}

    def test_nbfc_entity_types_list_covers_all_corpus_subtypes(self):
        # Guard: if the corpus gains new NBFC subtypes, update _NBFC_ENTITY_TYPES
        assert set(_NBFC_ENTITY_TYPES) >= {"NBFC-ICC", "NBFC-HFC", "NBFC-P2P", "NBFC-MFI"}

    # -- Cooperative bank entity_type filter ----------------------------------

    def test_cooperative_bank_uses_dollar_in(self):
        where, _, __ = _extract_filters("co-operative bank KYC violations")
        assert where == {"entity_type": {"$in": _COOP_ENTITY_TYPES}}

    def test_coop_entity_types_list_covers_corpus_subtypes(self):
        assert set(_COOP_ENTITY_TYPES) >= {"Bank-Cooperative", "Bank-UCB", "Bank-DCCB", "Bank-SCB"}

    # -- CoR filter removed -- data gap documented ----------------------------

    def test_cor_query_returns_no_chroma_filter(self):
        # CoR cancellation records don't exist in the vector store
        # (all bulk-excluded). Filter would always return 0 results.
        # Stage 3 (graph) handles CoR queries.
        where, _, __ = _extract_filters("Which NBFCs had their CoR cancelled?")
        # chroma_where should only have entity_type, no action_type
        if where is not None:
            keys = str(where)
            assert "cor_cancellation" not in keys
            assert "action_type" not in keys

    # -- Combined filters -----------------------------------------------------

    def test_year_plus_nbfc_date_separate_from_chroma(self):
        where, df, dt = _extract_filters("Which NBFCs were penalised in 2024?")
        assert where == {"entity_type": {"$in": _NBFC_ENTITY_TYPES}}
        assert df == "2024-01-01" and dt == "2025-01-01"

    # -- Named entity filter (Stage 4 addition) --------------------------------

    def test_named_entity_returns_entity_name_filter(self):
        # HDFC Bank is in the registry; entity_name filter must fire
        where, df, dt = _extract_filters("What penalty did RBI impose on HDFC Bank?")
        assert "entity_name" in where
        assert df is None and dt is None

    def test_named_entity_includes_all_stored_variants(self):
        # HDFC Bank has 3 canonical variants; all must appear in $in
        where, _, __ = _extract_filters("HDFC Bank KYC violation")
        assert "$in" in str(where)
        assert "HDFC Bank Limited" in str(where)

    def test_named_entity_with_year_combines_both_filters(self):
        # entity_name filter (ChromaDB) + date post-filter (Python) together
        where, df, dt = _extract_filters("What violations did HDFC Bank commit in 2023?")
        assert "entity_name" in where
        assert df == "2023-01-01" and dt == "2024-01-01"

    def test_alias_sbi_resolves_to_canonical(self):
        where, _, __ = _extract_filters("SBI penalties in 2023")
        assert "State Bank of India" in str(where)

    def test_multi_entity_uses_dollar_in(self):
        where, _, __ = _extract_filters("Penalties on Axis Bank and Yes Bank")
        assert "$in" in str(where)
        assert "Axis Bank Limited" in str(where)
        assert "Yes Bank Limited" in str(where)

    def test_named_entity_overrides_nbfc_entity_type(self):
        # When a named entity is detected, entity_type filter must NOT fire
        where, _, __ = _extract_filters("Axis Bank NBFC penalty")
        assert "entity_name" in str(where)
        assert "entity_type" not in str(where)

    # -- No filter cases -------------------------------------------------------

    def test_no_filter_generic_aggregation(self):
        # No named entity, no NBFC/coop keyword -> no ChromaDB filter
        where, df, dt = _extract_filters("What is the total penalty imposed by RBI?")
        assert where is None and df is None and dt is None

    def test_no_filter_empty_string(self):
        where, df, dt = _extract_filters("")
        assert where is None and df is None and dt is None

    def test_single_condition_no_and_wrapper(self):
        where, _, __ = _extract_filters("NBFC penalty")
        assert "$and" not in where

    # -- Known regex limitations (documented xfail) ---------------------------

    @pytest.mark.xfail(reason="Regex cannot parse relative time expressions")
    def test_paraphrase_last_year(self):
        _, df, dt = _extract_filters("Banks penalised last year")
        assert df is not None

    @pytest.mark.xfail(reason="Regex cannot detect negation")
    def test_negation_not_an_nbfc(self):
        where, _, __ = _extract_filters("Entities that are not NBFCs")
        assert where is None


# ── TestApplyDateFilter ───────────────────────────────────────────────────────

class TestApplyDateFilter:
    """Tests for _apply_date_filter() -- the Python post-filter step."""

    _CHUNKS = [
        _fake_chunk(date="2022-06-01", entity_name="A"),
        _fake_chunk(date="2023-04-06", entity_name="B"),
        _fake_chunk(date="2023-11-30", entity_name="C"),
        _fake_chunk(date="2024-01-15", entity_name="D"),
    ]

    def test_filters_by_year(self):
        result = _apply_date_filter(self._CHUNKS, "2023-01-01", "2024-01-01")
        assert [c["entity_name"] for c in result] == ["B", "C"]

    def test_only_date_from(self):
        result = _apply_date_filter(self._CHUNKS, "2023-06-01", None)
        assert [c["entity_name"] for c in result] == ["C", "D"]

    def test_only_date_to(self):
        result = _apply_date_filter(self._CHUNKS, None, "2023-01-01")
        assert [c["entity_name"] for c in result] == ["A"]

    def test_no_dates_returns_all(self):
        assert _apply_date_filter(self._CHUNKS, None, None) == self._CHUNKS

    def test_empty_result_when_no_match(self):
        assert _apply_date_filter(self._CHUNKS, "2020-01-01", "2021-01-01") == []


# ── TestAnswerPipeline ────────────────────────────────────────────────────────

class TestAnswerPipeline:
    """Pipeline behaviour tests. All external I/O mocked."""

    _CFG = {
        "models":       {"system_model": "claude-haiku-4-5-20251001"},
        "llm":          {"chat_temperature": 0, "hyde_temperature": 0.3,
                         "max_tokens_chat": 512},
        "advanced_rag": {"top_k_initial": 10, "top_k_final": 3},
        "embedding":    {"model": "nomic-embed-text",
                         "ollama": {"request_timeout": 30}},
        "chroma":       {"collection_name": "rbi_enforcement"},
    }

    def test_empty_retrieval_returns_graceful_answer(self):
        with patch("rag.advanced_rag.vector_retrieve", return_value=[]), \
             patch("rag.advanced_rag._generate_hypothesis", return_value="hyp"):
            result = answer("test question", cfg=self._CFG)
        assert "don't have enough information" in result["answer"].lower()
        assert result["contexts"] == [] and result["stage"] == "advanced"

    def test_hyde_failure_falls_back_to_raw_question(self):
        fake = [_fake_chunk()]
        with patch("rag.advanced_rag._generate_hypothesis",
                   side_effect=Exception("API down")), \
             patch("rag.advanced_rag.vector_retrieve",
                   return_value=fake) as mock_ret, \
             patch("rag.advanced_rag.rerank", return_value=fake), \
             patch("rag.advanced_rag._synthesise", return_value="ans"):
            result = answer("test question", cfg=self._CFG)
        assert mock_ret.call_args[0][0] == "test question"
        assert result["hypothesis"] == "test question"

    def test_reranker_failure_falls_back_to_cosine_top_k(self):
        fake = [_fake_chunk(entity_name=f"Bank {i}", score=0.9 - i * 0.05)
                for i in range(10)]
        top_k = self._CFG["advanced_rag"]["top_k_final"]
        with patch("rag.advanced_rag._generate_hypothesis", return_value="hyp"), \
             patch("rag.advanced_rag.vector_retrieve", return_value=fake), \
             patch("rag.advanced_rag.rerank",
                   side_effect=Exception("model load failed")), \
             patch("rag.advanced_rag._synthesise", return_value="ans"):
            result = answer("test question", cfg=self._CFG)
        assert len(result["contexts"]) == top_k

    def test_chroma_where_uses_dollar_in_for_nbfc(self):
        fake = [_fake_chunk(entity_type="NBFC-ICC")]
        with patch("rag.advanced_rag._generate_hypothesis", return_value="hyp"), \
             patch("rag.advanced_rag.vector_retrieve",
                   return_value=fake) as mock_ret, \
             patch("rag.advanced_rag.rerank", return_value=fake), \
             patch("rag.advanced_rag._synthesise", return_value="ans"):
            answer("Which NBFCs were penalised?", cfg=self._CFG)
        where = mock_ret.call_args[1]["where"]
        assert where == {"entity_type": {"$in": _NBFC_ENTITY_TYPES}}

    def test_chroma_where_uses_entity_name_for_named_institution(self):
        # Named institution -> entity_name $in filter, NOT entity_type
        fake = [_fake_chunk(entity_name="Axis Bank Limited")]
        with patch("rag.advanced_rag._generate_hypothesis", return_value="hyp"), \
             patch("rag.advanced_rag.vector_retrieve",
                   return_value=fake) as mock_ret, \
             patch("rag.advanced_rag.rerank", return_value=fake), \
             patch("rag.advanced_rag._synthesise", return_value="ans"):
            answer("What penalty did RBI impose on Axis Bank?", cfg=self._CFG)
        where = mock_ret.call_args[1]["where"]
        assert where is not None
        assert "entity_name" in str(where)
        assert "entity_type" not in str(where)
        assert "Axis Bank Limited" in str(where)

    def test_no_filter_for_generic_query(self):
        # Generic query without named entity -> where=None (full corpus scan)
        fake = [_fake_chunk()]
        with patch("rag.advanced_rag._generate_hypothesis", return_value="hyp"), \
             patch("rag.advanced_rag.vector_retrieve",
                   return_value=fake) as mock_ret, \
             patch("rag.advanced_rag.rerank", return_value=fake), \
             patch("rag.advanced_rag._synthesise", return_value="ans"):
            answer("Which banks violated KYC directions?", cfg=self._CFG)
        where = mock_ret.call_args[1]["where"]
        assert where is None

    def test_date_post_filtered_in_python(self):
        inside  = _fake_chunk(entity_name="Inside",  date="2023-06-01")
        outside = _fake_chunk(entity_name="Outside", date="2024-03-01")
        with patch("rag.advanced_rag._generate_hypothesis", return_value="hyp"), \
             patch("rag.advanced_rag.vector_retrieve", return_value=[inside, outside]), \
             patch("rag.advanced_rag.rerank", side_effect=lambda q, chunks, cfg: chunks), \
             patch("rag.advanced_rag._synthesise", return_value="ans"):
            result = answer("penalties in 2023", cfg=self._CFG)
        assert result["contexts"] == [inside["text"]]

    def test_result_has_required_keys(self):
        fake = [_fake_chunk()]
        with patch("rag.advanced_rag._generate_hypothesis", return_value="hyp"), \
             patch("rag.advanced_rag.vector_retrieve", return_value=fake), \
             patch("rag.advanced_rag.rerank", return_value=fake), \
             patch("rag.advanced_rag._synthesise", return_value="ans"):
            result = answer("test", cfg=self._CFG)
        assert {"question", "answer", "contexts", "hypothesis",
                "filters", "stage", "model"}.issubset(result.keys())
        assert {"chroma_where", "date_from", "date_to"}.issubset(result["filters"].keys())
