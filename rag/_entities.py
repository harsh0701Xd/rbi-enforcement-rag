"""
rag/_entities.py

Single source of truth for named entity detection and entity-type taxonomy
used across the RAG pipeline.

Used by:
  - rag/nodes/router.py      : _has_named_entity() for routing decisions
  - rag/advanced_rag.py      : _extract_filters() for ChromaDB pre-filtering
  - rag/nodes/retrieval.py   : vector_node for agentic pre-filtering (Stage 4)

Design notes:
  ENTITY_REGISTRY maps lowercase query phrase -> list of canonical entity_name
  strings exactly as stored in ChromaDB metadata.  Multiple variants per key
  ("Limited" vs "Ltd") are handled via $in.  Keys sorted longest-first in the
  compiled regex so longer phrases (e.g. "central bank of india") are matched
  before shorter substrings (e.g. "bank of india"), preventing false positives.

  _ENTITY_FALLBACK_RE catches long-tail institution names not in the registry
  (e.g. "Xyz Co-operative Bank") but returns no canonical form, so it is used
  for routing only -- never for ChromaDB filtering.

Public API:
  ENTITY_REGISTRY          dict[str, list[str]]
  NBFC_ENTITY_TYPES        list[str]   -- all NBFC entity_type values in corpus
  COOP_ENTITY_TYPES        list[str]   -- all cooperative bank entity_type values
  extract_entities(text)   -> list[str]  canonical names for ChromaDB $in filter
  has_named_entity(text)   -> bool       True if any institution name detected
"""

from __future__ import annotations

import re

# ── Entity type taxonomy ──────────────────────────────────────────────────────
# All entity_type values observed in the corpus.  Single source of truth --
# imported by advanced_rag._extract_filters() and prompts/ that need the list.

NBFC_ENTITY_TYPES: list[str] = [
    "NBFC-ICC",
    "NBFC-HFC",
    "NBFC-P2P",
    "NBFC-MFI",
]

COOP_ENTITY_TYPES: list[str] = [
    "Bank-Cooperative",
    "Bank-UCB",
    "Bank-DCCB",
    "Bank-SCB",
]

ALL_ENTITY_TYPES: list[str] = [
    "Bank-Private", "Bank-Public", "Bank-UCB", "Bank-Cooperative",
    "Bank-DCCB", "Bank-SCB", "Bank-RRB", "Bank-SFB", "Bank-Payment",
    "NBFC-P2P", "NBFC-ICC", "NBFC-HFC", "NBFC-MFI",
    "ARC", "CIC", "PA", "PPI", "UNKNOWN",
]

# ── Named entity registry ─────────────────────────────────────────────────────
# Keys  : lowercase phrases as they appear in user queries (including aliases).
# Values: list of entity_name strings exactly as stored in ChromaDB metadata.
#         Use $in for the filter so all variants are covered.
#
# Absent from registry (not in corpus):
#   au small finance bank, utkarsh small finance bank, fincare small finance bank,
#   airtel payments bank, nsdl payments bank, tata capital, aditya birla finance,
#   piramal finance, cholamandalam, fullerton india, incred finance,
#   lakshmi vilas bank (merged -> DBS), bank of rajasthan (merged -> ICICI),
#   punjab and maharashtra cooperative bank / pmc bank (wound down)

ENTITY_REGISTRY: dict[str, list[str]] = {

    # ── Private sector banks ──────────────────────────────────────────────────
    "axis bank":                ["Axis Bank Limited", "Axis Bank Ltd"],
    "hdfc bank":                ["HDFC Bank Limited",
                                 "Authorised Dealer Bank – HDFC Bank Limited",
                                 "Authorised Dealer Bank – HDFC Bank Ltd"],
    "icici bank":               ["ICICI Bank Limited", "ICICI Bank Ltd"],
    "kotak mahindra bank":      ["Kotak Mahindra Bank Limited"],
    "kotak bank":               ["Kotak Mahindra Bank Limited"],           # alias
    "yes bank":                 ["Yes Bank Limited", "Yes Bank Ltd"],
    "indusind bank":            ["IndusInd Bank Limited", "IndusInd Bank Ltd"],
    "federal bank":             ["Federal Bank Limited", "Federal Bank Ltd"],
    "rbl bank":                 ["RBL Bank Limited"],
    "south indian bank":        ["South Indian Bank Limited"],
    "city union bank":          ["City Union Bank Limited"],
    "karur vysya bank":         ["Karur Vysya Bank Limited", "Karur Vysya Bank Ltd"],
    "bandhan bank":             ["Bandhan Bank Limited", "Bandhan Bank Ltd"],
    "dcb bank":                 ["DCB Bank Limited", "DCB Bank Ltd"],
    "tamilnad mercantile bank": ["Tamilnad Mercantile Bank Limited",
                                 "Tamilnad Mercantile Bank Ltd"],
    "jammu and kashmir bank":   ["Jammu and Kashmir Bank Limited"],
    "j&k bank":                 ["Jammu and Kashmir Bank Limited"],         # alias
    "dhanlaxmi bank":           ["Dhanlaxmi Bank Ltd"],
    "nainital bank":            ["The Nainital Bank Limited",
                                 "The Nainital Bank Ltd"],
    "csb bank":                 ["CSB Bank Limited", "CSB Bank Ltd"],
    "catholic syrian bank":     ["CSB Bank Limited", "CSB Bank Ltd"],      # former name

    # ── Public sector banks ───────────────────────────────────────────────────
    "state bank of india":      ["State Bank of India"],
    "sbi":                      ["State Bank of India"],                    # alias
    "bank of baroda":           ["Bank of Baroda"],
    "punjab national bank":     ["Punjab National Bank"],
    "pnb":                      ["Punjab National Bank"],                   # alias
    "canara bank":              ["Canara Bank"],
    "indian bank":              ["Indian Bank"],
    "union bank of india":      ["Union Bank of India"],
    "central bank of india":    ["Central Bank of India"],
    "uco bank":                 ["UCO Bank"],
    "bank of india":            ["Bank of India"],
    "bank of maharashtra":      ["Bank of Maharashtra"],
    "punjab and sind bank":     ["Punjab and Sind Bank", "Punjab & Sind Bank"],
    "indian overseas bank":     ["Indian Overseas Bank"],
    "idbi bank":                ["IDBI Bank Limited"],
    "idfc first bank":          ["IDFC First Bank Limited"],

    # ── Small finance banks ───────────────────────────────────────────────────
    "equitas small finance bank":    ["Equitas Small Finance Bank Limited"],
    "ujjivan small finance bank":    ["Ujjivan Small Finance Bank Limited"],
    "jana small finance bank":       ["Jana Small Finance Bank Limited"],
    "suryoday small finance bank":   ["Suryoday Small Finance Bank Ltd"],
    "esaf small finance bank":       ["ESAF Small Finance Bank Limited"],
    "north east small finance bank": ["North East Small Finance Bank Limited"],

    # ── Payment banks ─────────────────────────────────────────────────────────
    "india post payments bank":  ["India Post Payments Bank Limited"],
    "paytm payments bank":       ["Paytm Payments Bank Limited"],
    "fino payments bank":        ["Fino Payments Bank Limited"],
    "jio payments bank":         ["Jio Payments Bank Limited"],

    # ── Major NBFCs ───────────────────────────────────────────────────────────
    "bajaj finance":             ["Bajaj Finance Limited"],
    "muthoot finance":           ["Muthoot Finance Limited"],
    "manappuram finance":        ["Manappuram Finance Limited"],
    "shriram finance":           ["Shriram Finance Limited"],
    "l&t finance":               ["L&T Finance Limited"],
    "hero fincorp":              ["Hero FinCorp Limited"],
    "mahindra finance":          ["Mahindra & Mahindra Financial Services Limited"],
    "mahindra & mahindra financial services": ["Mahindra & Mahindra Financial Services Limited"],
    "housing development finance corporation": ["Housing Development Finance Corporation Limited"],
    "hdfc limited":              ["Housing Development Finance Corporation Limited"],
    "hdfc ltd":                  ["Housing Development Finance Corporation Limited"],

    # ── Cooperative / UCB ─────────────────────────────────────────────────────
    "saraswat co-operative bank":    ["Saraswat Co-operative Bank Limited",
                                      "Saraswat Co-operative Bank Ltd"],
    "saraswat bank":                 ["Saraswat Co-operative Bank Limited",
                                      "Saraswat Co-operative Bank Ltd"],
    "cosmos co-operative bank":      ["The Cosmos Co-operative Bank Ltd"],
    "cosmos bank":                   ["The Cosmos Co-operative Bank Ltd"],
    "abhyudaya co-operative bank":   ["Abhyudaya Co-operative Bank Limited"],
    "abhyudaya bank":                ["Abhyudaya Co-operative Bank Limited"],
    "apna sahakari bank":            ["Apna Sahakari Bank Ltd"],
    "new india co-operative bank":   ["New India Co-operative Bank Limited",
                                      "New India Co-operative Bank Ltd"],
    "new india cooperative bank":    ["New India Co-operative Bank Limited",
                                      "New India Co-operative Bank Ltd"],
}

# ── Compiled regex ────────────────────────────────────────────────────────────
# Keys sorted longest-first so "central bank of india" matches before
# "bank of india", and "south indian bank" matches before "indian bank".

_REGISTRY_RE = re.compile(
    r'\b(' + '|'.join(
        re.escape(k)
        for k in sorted(ENTITY_REGISTRY, key=len, reverse=True)
    ) + r')\b',
    re.IGNORECASE,
)

# Fallback: catches long-tail institution names not in the registry.
# Used for routing ONLY -- returns no canonical form so never used for
# ChromaDB filtering (we don't know the stored entity_name for unknowns).
_ENTITY_FALLBACK_RE = re.compile(
    r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+'
    r'(?:Bank|Finance|Capital|Credit|NBFC|Grameen|Microfinance|Co-?operative)\b',
    re.IGNORECASE,
)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_entities(text: str) -> list[str]:
    """
    Return canonical entity_name strings (as stored in ChromaDB) for all
    named institutions mentioned in *text*.

    Suitable for use as a ChromaDB $in filter:
        {"entity_name": {"$in": extract_entities(question)}}

    Returns an empty list if no registry entry matches.  The fallback regex
    is NOT used here -- unknown institutions have no canonical form.
    """
    matches = _REGISTRY_RE.findall(text)
    canonical: list[str] = []
    seen: set[str] = set()
    for m in matches:
        for variant in ENTITY_REGISTRY[m.lower()]:
            if variant not in seen:
                seen.add(variant)
                canonical.append(variant)
    return canonical


def has_named_entity(text: str) -> bool:
    """
    Return True if *text* mentions any specific regulated institution.
    Uses registry first, then the fallback pattern for long-tail names.
    Intended for routing decisions (router.py).
    """
    if _REGISTRY_RE.search(text):
        return True
    return bool(_ENTITY_FALLBACK_RE.search(text))
