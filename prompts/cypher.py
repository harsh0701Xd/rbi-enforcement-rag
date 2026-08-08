"""
prompts/cypher.py

NL → Cypher generation prompts for Stage 3 graph retrieval.
Must stay in sync with ingestion/graph_writer.py and _rbi_graph_queries.py.

Exports:
  SCHEMA  -- RBI enforcement graph schema injected verbatim into every prompt
  PROMPT  -- Cypher generation prompt; format with {schema}, {limit}, {question}
"""

# ── Graph schema ──────────────────────────────────────────────────────────────
# Mirrors Schema A written by ingestion/graph_writer.py exactly.

SCHEMA = """\
Node labels and properties:
  Institution  : name (str), entity_type (str)
  PenaltyAction: prid (int), date (str "YYYY-MM-DD"), title (str),
                 penalty_lakh (float or null), action_type (str),
                 source_url (str), pr_number (str)
  Violation    : text_key (str), text (str)
  Regulation   : citation_key (str), citation (str)

Relationships (directed arrows):
  (Institution)  -[:RECEIVED]      -> (PenaltyAction)
  (PenaltyAction)-[:FOR_VIOLATION] -> (Violation)
  (PenaltyAction)-[:CITES]         -> (Regulation)

entity_type values in the corpus:
  Banks : "Bank-Private", "Bank-Public", "Bank-Cooperative", "Bank-UCB",
          "Bank-DCCB", "Bank-SCB", "Bank-RRB", "Bank-SFB", "Bank-Payment"
  NBFCs : "NBFC-ICC", "NBFC-HFC", "NBFC-P2P", "NBFC-MFI"
  Other : "PPI", "CIC", "PA", "ARC"

action_type values: "monetary_penalty" | "other"

Date range example (date is stored as "YYYY-MM-DD" string, lexicographic ok):
  WHERE p.date >= "2023-01-01" AND p.date < "2024-01-01\""""


# ── Cypher generation prompt ──────────────────────────────────────────────────

PROMPT = """\
You are a Neo4j Cypher expert. Generate ONE Cypher query that answers the question.

GRAPH SCHEMA:
{schema}

STRICT RULES — violating any rule produces unusable output:
1. Always include LIMIT {limit}, UNLESS the query requires aggregation (see Rule 12)
2. RETURN only properties, never full nodes  →  RETURN i.name, p.date   not   RETURN i
3. Use OPTIONAL MATCH for relationships that may not exist
4. Every MATCH clause must include a label  →  MATCH (i:Institution)   not   MATCH (i)
5. Use case-insensitive name matching       →  WHERE toLower(i.name) CONTAINS toLower("axis bank")
6. For NBFC queries use $in syntax          →  WHERE i.entity_type IN ["NBFC-ICC","NBFC-HFC","NBFC-P2P","NBFC-MFI"]
7. For date ranges use string comparison    →  WHERE p.date >= "2023-01-01" AND p.date < "2024-01-01"
8. Order by date descending when relevant   →  ORDER BY p.date DESC
9. If you traverse to Violation nodes (v), always include v.text in RETURN
   so the synthesiser can read the violation description
10. If you traverse to Regulation nodes (r), always include r.citation in RETURN
11. Output ONLY the raw Cypher — no explanation, no markdown, no code fences
12. AGGREGATION RULE — if the question starts with [AGGREGATION]: use Cypher aggregate
    functions (COUNT, SUM, AVG) and omit LIMIT entirely.
    Examples:
      COUNT: MATCH (i:Institution)-[:RECEIVED]->(p:PenaltyAction)
             WHERE p.date >= "2023-01-01" AND p.date < "2024-01-01"
             RETURN COUNT(DISTINCT i.name) AS institution_count
      SUM:   MATCH (i:Institution)-[:RECEIVED]->(p:PenaltyAction)
             WHERE p.penalty_lakh IS NOT NULL
             RETURN SUM(p.penalty_lakh) AS total_penalty_lakh

Question: {question}
Cypher:"""
