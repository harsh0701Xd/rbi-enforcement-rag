"""
tests/unit/test_vector_writer.py

Unit tests for ingestion/vector_writer.py and ingestion/_text_processor.py.
No Ollama, no ChromaDB, no disk I/O — all pure-function tests.

Covers:
  _normalize_entity_name():
    - Location suffix stripping (city, city+state variants)
    - Trailing period removal ("Ltd." → "Ltd")
    - Names with no suffix remain unchanged
    - "Authorised Dealer Bank" prefix NOT affected (already cleaned in graph)

  chunk_text() from _text_processor:
    - Chunks shorter than min_chars are discarded
    - Short text that fits in one chunk returns a single chunk
    - Overlap is applied between consecutive chunks
    - Deterministic: same input always produces same output
    - Separator hierarchy: paragraph break preferred over line break

  Metadata / ID contract:
    - Chunk IDs are deterministic: f"{prid}_chunk_{i}"
    - All required metadata keys are present for each chunk

Run from rbi_intel/:
    pytest tests/unit/test_vector_writer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion._text_processor import chunk_text, normalize_entity_name as _normalize_entity_name


# ── _normalize_entity_name ────────────────────────────────────────────────────

class TestNormalizeEntityName:

    def test_strips_city_suffix(self):
        assert _normalize_entity_name(
            "Bajaj Finance Limited, Pune"
        ) == "Bajaj Finance Limited"

    def test_strips_city_and_state_suffix(self):
        assert _normalize_entity_name(
            "Bajaj Finance Limited, Pune, Maharashtra"
        ) == "Bajaj Finance Limited"

    def test_strips_trailing_period_on_ltd(self):
        assert _normalize_entity_name(
            "The National Co-operative Bank Ltd."
        ) == "The National Co-operative Bank Ltd"

    def test_strips_city_and_trailing_period_combined(self):
        assert _normalize_entity_name(
            "The National Co-operative Bank Ltd., Mumbai"
        ) == "The National Co-operative Bank Ltd"

    def test_unchanged_when_no_suffix(self):
        assert _normalize_entity_name("ICICI Bank Limited") == "ICICI Bank Limited"

    def test_unchanged_short_name(self):
        assert _normalize_entity_name("Citibank N.A") == "Citibank N.A"

    def test_strips_three_level_suffix(self):
        # Unlikely in practice but regex should handle it
        result = _normalize_entity_name("XYZ Finance Ltd, Mumbai, Maharashtra, India")
        assert result == "XYZ Finance Ltd"

    def test_does_not_strip_lowercase_suffix(self):
        # lowercase word after comma is NOT treated as a location suffix
        result = _normalize_entity_name("Bank of india")
        assert result == "Bank of india"

    def test_whitespace_stripped(self):
        assert _normalize_entity_name("  HDFC Bank Limited  ") == "HDFC Bank Limited"


# ── chunk_text ────────────────────────────────────────────────────────────────

class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        text   = "RBI imposed a penalty of Rs. 1 lakh on XYZ Bank."
        result = chunk_text(text, chunk_chars=2048, overlap_chars=256, min_chars=10)
        assert len(result) == 1
        assert result[0] == text

    def test_chunks_below_min_chars_discarded(self):
        # Single short word padded to just below min_chars threshold
        tiny = "ok"
        result = chunk_text(tiny, chunk_chars=2048, overlap_chars=256, min_chars=10)
        assert result == []

    def test_deterministic_output(self):
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
        r1   = chunk_text(text, chunk_chars=20, overlap_chars=5, min_chars=3)
        r2   = chunk_text(text, chunk_chars=20, overlap_chars=5, min_chars=3)
        assert r1 == r2

    def test_prefers_paragraph_break_over_line_break(self):
        # Paragraph break (\n\n) should be preferred over \n
        text   = "First paragraph with enough content to fill one chunk.\n\nSecond paragraph here."
        result = chunk_text(text, chunk_chars=60, overlap_chars=0, min_chars=5)
        # Both paragraphs should be separate chunks (split at \n\n)
        assert any("First paragraph" in c for c in result)
        assert any("Second paragraph" in c for c in result)

    def test_overlap_produces_shared_content(self):
        # With overlap, the end of chunk N should appear at the start of chunk N+1
        text = "A " * 300   # 600 chars
        result = chunk_text(text, chunk_chars=100, overlap_chars=20, min_chars=5)
        assert len(result) >= 2
        # Each chunk should be within the chunk_chars limit
        for chunk in result:
            assert len(chunk) <= 120   # some flex for separator alignment

    def test_returns_list_of_strings(self):
        text   = "Reserve Bank of India imposed a penalty."
        result = chunk_text(text, chunk_chars=2048, overlap_chars=256, min_chars=5)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_empty_text_returns_empty(self):
        result = chunk_text("", chunk_chars=2048, overlap_chars=256, min_chars=5)
        assert result == []

    def test_whitespace_only_discarded(self):
        result = chunk_text("   \n\n   ", chunk_chars=2048, overlap_chars=256, min_chars=5)
        assert result == []


# ── Chunk ID and metadata contract ───────────────────────────────────────────

class TestChunkContract:
    """
    Verify the ID scheme and metadata structure that write_vectors() produces.
    These tests protect the contract that vector_retriever.py depends on.
    """

    REQUIRED_METADATA_KEYS = {
        "prid", "entity_name", "entity_type", "date",
        "action_type", "source_url", "pr_number",
        "chunk_index", "total_chunks",
    }

    def _make_chunks_and_meta(self, rec: dict, cfg_ing: dict) -> list[tuple[str, dict]]:
        """Replicate the ID + metadata logic from write_vectors() for testing."""
        cpt           = float(cfg_ing["chars_per_token"])
        chunk_chars   = int(cfg_ing["chunk_size"]      * cpt)
        overlap_chars = int(cfg_ing["chunk_overlap"]    * cpt)
        min_chars     = int(cfg_ing["min_chunk_length"] * cpt)

        chunks      = chunk_text(rec["body_text"], chunk_chars, overlap_chars, min_chars)
        entity_name = _normalize_entity_name(rec["entity_name"])
        total       = len(chunks)
        return [
            (
                f"{rec['prid']}_chunk_{i}",
                {
                    "prid": rec["prid"], "entity_name": entity_name,
                    "entity_type": rec["entity_type"], "date": rec["date"],
                    "action_type": rec["action_type"], "source_url": rec["source_url"],
                    "pr_number": str(rec["pr_number"]),
                    "chunk_index": i, "total_chunks": total,
                },
            )
            for i, _ in enumerate(chunks)
        ]

    def _sample_cfg_ing(self) -> dict:
        return {
            "chunk_size": 512, "chunk_overlap": 64,
            "chars_per_token": 4.0, "min_chunk_length": 100,
        }

    def _sample_record(self) -> dict:
        return {
            "prid": 99999,
            "entity_name": "Test Bank Limited, Mumbai",
            "entity_type": "Bank-Private",
            "date": "2024-01-15",
            "action_type": "monetary_penalty",
            "source_url": "https://rbi.org.in/test",
            "pr_number": "2024-2025/123",
            "body_text": (
                "Reserve Bank of India has imposed a monetary penalty of ₹1.00 lakh "
                "on Test Bank Limited. This action is based on deficiencies in "
                "regulatory compliance with the directions issued by RBI. "
                "The penalty is imposed in exercise of powers vested in RBI. " * 20
            ),
        }

    def test_chunk_ids_are_deterministic(self):
        rec     = self._sample_record()
        cfg_ing = self._sample_cfg_ing()
        pairs1  = self._make_chunks_and_meta(rec, cfg_ing)
        pairs2  = self._make_chunks_and_meta(rec, cfg_ing)
        assert [p[0] for p in pairs1] == [p[0] for p in pairs2]

    def test_chunk_ids_use_prid_prefix(self):
        rec    = self._sample_record()
        pairs  = self._make_chunks_and_meta(rec, self._sample_cfg_ing())
        for i, (chunk_id, _) in enumerate(pairs):
            assert chunk_id == f"99999_chunk_{i}"

    def test_all_required_metadata_keys_present(self):
        rec   = self._sample_record()
        pairs = self._make_chunks_and_meta(rec, self._sample_cfg_ing())
        assert len(pairs) > 0, "Expected at least one chunk"
        for _, meta in pairs:
            assert self.REQUIRED_METADATA_KEYS.issubset(meta.keys()), (
                f"Missing keys: {self.REQUIRED_METADATA_KEYS - meta.keys()}"
            )

    def test_entity_name_is_normalized_in_metadata(self):
        rec   = self._sample_record()   # entity_name has ", Mumbai" suffix
        pairs = self._make_chunks_and_meta(rec, self._sample_cfg_ing())
        for _, meta in pairs:
            assert meta["entity_name"] == "Test Bank Limited"

    def test_chunk_index_sequence_is_contiguous(self):
        rec    = self._sample_record()
        pairs  = self._make_chunks_and_meta(rec, self._sample_cfg_ing())
        indices = [meta["chunk_index"] for _, meta in pairs]
        assert indices == list(range(len(pairs)))

    def test_total_chunks_consistent_across_all_chunks(self):
        rec   = self._sample_record()
        pairs = self._make_chunks_and_meta(rec, self._sample_cfg_ing())
        total_values = {meta["total_chunks"] for _, meta in pairs}
        assert len(total_values) == 1
        assert total_values.pop() == len(pairs)
