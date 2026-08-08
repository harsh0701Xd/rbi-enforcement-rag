"""
ingestion/_text_processor.py

Text processing utilities for the RBI enforcement corpus.
Used by vector_writer.py and retrieval modules.

Public API:
    OllamaUnavailableError      -- sentinel exception for upstream callers
    normalize_entity_name(name) -> str
    chunk_text(text, ...)       -> list[str]
"""

from __future__ import annotations

import re


# -- Sentinel exception --------------------------------------------------------

class OllamaUnavailableError(RuntimeError):
    """
    Raised when the Ollama embedding service cannot be reached.
    Callers must NOT catch this silently -- surface it so the operator knows
    to run `ollama serve` before retrying.
    """


# -- Entity name normalisation -------------------------------------------------

_LOCATION_SUFFIX_RE = re.compile(r",\s+[A-Z][^,]*(?:,\s+[A-Z][^,]*)*$")


def normalize_entity_name(name: str) -> str:
    """
    Strip location suffix and trailing punctuation from institution names.
    Mirrors Category A cleanup applied during the graph audit.

    Examples:
        "Bajaj Finance Limited, Pune, Maharashtra" -> "Bajaj Finance Limited"
        "The National Co-operative Bank Ltd., Mumbai" -> "The National Co-operative Bank Ltd"
        "ICICI Bank Limited"  -> "ICICI Bank Limited"   (unchanged)
    """
    name = _LOCATION_SUFFIX_RE.sub("", name).strip()
    name = re.sub(r"\.$", "", name)      # "Ltd." -> "Ltd"
    return name


# -- Text to chunks ------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[str]:
    """
    Recursive character splitter -- no external dependencies.

    Splits preferring paragraph boundaries (\\n\\n), then line breaks,
    then sentence ends, then spaces. Strips each candidate and discards
    chunks shorter than min_chars.

    Args:
        text:          clean plain text (e.g. body_text from an EnforcementRecord)
        chunk_chars:   maximum chunk length in characters
        overlap_chars: characters of overlap between consecutive chunks
        min_chars:     discard chunks shorter than this (noisy fragments)

    Returns list of non-overlapping string chunks ready for embedding.
    """
    if len(text) <= chunk_chars:
        return [text.strip()] if len(text.strip()) >= min_chars else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))

        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                pos = text.rfind(sep, start, end)
                if pos > start:
                    end = pos + len(sep)
                    break

        candidate = text[start:end].strip()
        if len(candidate) >= min_chars:
            chunks.append(candidate)

        next_start = end - overlap_chars
        if next_start > start:
            start = next_start
        else:
            start = end

    return chunks
