"""
prompts/extraction.py

LLM extraction prompt for ingestion/extractor.py.

The prompt is built via string concatenation (not .format()) because the body
text of RBI press releases regularly contains literal { } characters that would
cause KeyError if .format() were used.

Exports:
  PROMPT_STATIC  -- the fixed prefix that precedes the per-record context
  build_prompt   -- assembles the full prompt; use this, never .format()
"""

# Literal { } in the JSON example are intentional schema illustration.
# Pure string concatenation ensures they are never interpreted as format slots.
PROMPT_STATIC = (
    "Extract structured data from this RBI enforcement press release.\n\n"
    "Return ONLY a JSON object — no prose, no markdown fences:\n"
    "{\n"
    '  "entity_type": "<one canonical type>",\n'
    '  "violations": ["<clause 1>", "<clause 2>"],\n'
    '  "regulation_refs": ["<citation 1>", "<citation 2>"]\n'
    "}\n\n"
    "Canonical entity types (choose exactly one):\n"
    "  Bank-Private, Bank-Public, Bank-UCB, Bank-Cooperative, Bank-DCCB, Bank-SCB,\n"
    "  Bank-RRB, Bank-SFB, Bank-Payment, NBFC-P2P, NBFC-ICC, NBFC-HFC, NBFC-MFI,\n"
    "  ARC, CIC, PA, PPI, UNKNOWN\n\n"
    "Rules:\n"
    "1. entity_type — If the current type is already known (not UNKNOWN), return it\n"
    "   unchanged. Otherwise classify the penalised institution from the body text.\n"
    '2. violations — Each item <= 30 words. Source phrases: "non-compliance with",\n'
    '   "failure to", "contravention of", or explicitly numbered items like (i) (ii).\n'
    "   Return [] if no violations are stated explicitly.\n"
    "3. regulation_refs — Specific statute + section only, e.g.\n"
    '   "Section 26A of Banking Regulation Act 1949" or\n'
    '   "RBI Master Directions on KYC 2016".\n'
    '   Exclude vague phrases like "applicable RBI guidelines". Return [] if none.\n\n'
)


def build_prompt(entity_name: str, entity_type: str, body_text: str) -> str:
    """
    Assemble the full extraction prompt via string concatenation.

    No .format() calls anywhere — entity_name, entity_type, and body_text
    may contain any characters (including { }) without risk of KeyError.
    """
    return (
        PROMPT_STATIC
        + "Entity name  : " + entity_name + "\n"
        + "Current type : " + entity_type + "\n\n"
        + "Body text:\n"
        + body_text
    )
