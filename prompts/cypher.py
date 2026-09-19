"""
prompts/cypher.py

NL → Cypher generation prompts for Stage 6 graph retrieval.
Must stay in sync with ingestion/graph_writer.py and _rbi_graph_queries.py.

Stage 6 changes:
  - PROMPT: added WITH alias rule, cooperative bank IN rule, COUNT disambiguation,
    standardized column aliases, trend/comparison/absence patterns
  - MULTIHOP_PROMPT: added WITH alias rule (fixes SyntaxError on chained queries)
  - Both prompts now use cypher_model (Sonnet) instead of system_model (Haiku)

Exports:
  SCHEMA          -- RBI enforcement graph schema injected verbatim into every prompt
  PROMPT          -- Single-hop Cypher generation; format with {schema}, {limit}, {question}
  MULTIHOP_PROMPT -- Multi-hop Cypher generation; format with {schema}, {limit}, {sub_questions}
                     Used by retrieve_multihop() when cypher_sufficient=True
"""

# ── Graph schema ──────────────────────────────────────────────────────────────
# Mirrors Schema A written by ingestion/graph_writer.py exactly.

SCHEMA = """\
Node labels and properties:
  Institution  : name (str), entity_type (str), state (str or null)
                 state uses a three-tier system:
                   - A state/UT name (e.g. "Maharashtra") for regional entities
                   - "National" for pan-India Indian entities (large banks, NBFCs)
                   - "Foreign" for non-Indian entities (foreign banks)
                   - null only for the two intentionally unresolved nodes
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
6. For NBFC queries use IN syntax           →  WHERE i.entity_type IN ["NBFC-ICC","NBFC-HFC","NBFC-P2P","NBFC-MFI"]
7. For date ranges use string comparison    →  WHERE p.date >= "2023-01-01" AND p.date < "2024-01-01"
8. Order by date descending when relevant   →  ORDER BY p.date DESC
9. Whenever a PenaltyAction node (p) is in scope, ALWAYS add BOTH:
     OPTIONAL MATCH (p)-[:FOR_VIOLATION]->(v:Violation)
     OPTIONAL MATCH (p)-[:CITES]->(r:Regulation)
   and include v.text AND r.citation AND p.pr_number in RETURN.
   v.text is the factual description of what the institution did wrong.
   r.citation is the legal/regulatory basis for the penalty.
   p.pr_number is required for inline citations.
   Include all three regardless of which aspect the question emphasises.
10. Output ONLY the raw Cypher — no explanation, no markdown, no code fences
11. WITH alias — every computed expression in a WITH clause MUST have an AS alias.
    Wrong:  WITH i, COUNT(p)
    Correct: WITH i, COUNT(p) AS n_penalties
    Wrong:  WITH toInteger(substring(p.date, 0, 4))
    Correct: WITH toInteger(substring(p.date, 0, 4)) AS year
12. AGGREGATION RULE — if the question starts with [AGGREGATION]: use Cypher aggregate
    functions (COUNT, SUM, AVG) and omit LIMIT entirely.
    Standardized aliases for aggregate returns:
      COUNT(DISTINCT i.name) AS institution_count   -- counting distinct institutions
      COUNT(DISTINCT p.prid) AS penalty_count       -- counting distinct penalty events
      SUM(p.penalty_lakh)    AS total_lakh
      AVG(p.penalty_lakh)    AS avg_lakh
    Examples:
      [AGGREGATION] how many banks:
        MATCH (i:Institution)-[:RECEIVED]->(p:PenaltyAction)
        WHERE p.date >= "2023-01-01" AND p.date < "2024-01-01"
        RETURN COUNT(DISTINCT i.name) AS institution_count
      [AGGREGATION] total penalty:
        MATCH (i:Institution)-[:RECEIVED]->(p:PenaltyAction)
        WHERE p.penalty_lakh IS NOT NULL
        RETURN SUM(p.penalty_lakh) AS total_lakh
13. COUNT DISAMBIGUATION — distinguish what you are counting:
    - "how many banks/institutions/NBFCs" → COUNT(DISTINCT i.name) AS institution_count
    - "how many penalties/cases/actions/times" → COUNT(DISTINCT p.prid) AS penalty_count
    Never use COUNT(DISTINCT p) to count institutions — p is a PenaltyAction node.
14. COOPERATIVE BANK RULE — "cooperative bank" covers multiple entity_type subtypes.
    Always use IN filter for cooperative bank queries:
      WHERE i.entity_type IN ["Bank-Cooperative","Bank-UCB","Bank-DCCB","Bank-SCB"]
    Never filter on entity_type = "Bank-Cooperative" alone — this misses UCBs, DCCBs, SCBs.
15. STANDARDIZED RETURN ALIASES — use these consistent column names for portability:
    i.name  AS entity_name     (institution name, used by {{q0.entity_name}} resolver)
    i.state AS state           (institution state/UT, used by {{q0.state}} resolver)
    p.date  AS p_date          (penalty date)
    p.penalty_lakh AS penalty_lakh
    toInteger(substring(p.date, 0, 4)) AS year   (used by {{q0.year}} resolver)
16. TREND PATTERN — for year-over-year queries, group by year:
    WITH toInteger(substring(p.date, 0, 4)) AS year,
         COUNT(DISTINCT i.name) AS institution_count,
         SUM(p.penalty_lakh) AS total_lakh
    ORDER BY year
    RETURN year, institution_count, total_lakh
17. COMPARISON PATTERN — for side-by-side entity comparisons, return one row per entity:
    MATCH (i:Institution)-[:RECEIVED]->(p:PenaltyAction)
    WHERE toLower(i.name) CONTAINS toLower("axis bank")
       OR toLower(i.name) CONTAINS toLower("hdfc bank")
    WITH i.name AS entity_name,
         COUNT(DISTINCT p.prid) AS penalty_count,
         SUM(p.penalty_lakh) AS total_lakh
    ORDER BY total_lakh DESC
    RETURN entity_name, penalty_count, total_lakh
19. STATE FILTER / GROUP BY — use i.state for state-level queries.
    Filter by state:
      WHERE i.state = "Maharashtra"
    Filter for foreign entities (foreign banks, non-Indian institutions):
      WHERE i.state = "Foreign"
    Filter for pan-India Indian entities:
      WHERE i.state = "National"
    Filter for regional (state-specific) entities only:
      WHERE i.state IS NOT NULL AND i.state <> "National" AND i.state <> "Foreign"
    Group by state (e.g. "how many cooperative banks per state"):
      WITH i.state AS state, COUNT(DISTINCT i.name) AS institution_count
      ORDER BY institution_count DESC
      RETURN state, institution_count
    Note: always add WHERE i.state IS NOT NULL before grouping to exclude the two
    intentionally unresolved null nodes from breakdowns.

18. ABSENCE PATTERN — for "which X were NOT penalised" queries:
    MATCH (i:Institution)
    WHERE NOT EXISTS {{
      MATCH (i)-[:RECEIVED]->(p:PenaltyAction)
      WHERE p.date >= "2023-01-01" AND p.date < "2024-01-01"
    }}
    RETURN i.name AS entity_name
    LIMIT {limit}

Question: {question}
Cypher:"""


# ── Multi-hop Cypher generation prompt ───────────────────────────────────────
# Used only when cypher_sufficient=True (all sub-questions answerable from graph).
# Generates a single WITH-chained Cypher covering all hops in one DB round-trip.

MULTIHOP_PROMPT = """\
You are a Neo4j Cypher expert. Generate ONE multi-hop Cypher query that answers
ALL of the sub-questions below in a single query using WITH to chain results.

GRAPH SCHEMA:
{schema}

SUB-QUESTIONS TO ANSWER (in order):
{sub_questions}

STRICT RULES:
1. Use WITH to chain results between hops — do not write multiple MATCH clauses
   that restart the traversal independently.
2. Two distinct patterns — do NOT mix them up:
   a) TOP-N LOOKUP: use LIMIT inside WITH to select the top entity, then join further.
      Example: find the bank with the highest penalty, then get its violations.
        WITH i, p.penalty_lakh AS penalty_lakh ORDER BY penalty_lakh DESC LIMIT 1
   b) GROUP BY AGGREGATION: "for each X [across ALL X], count/sum Y" — do NOT use LIMIT.
      Use WITH to group and aggregate across all values, then RETURN all groups.
      Example: for each violation type, count banks:
        WITH v.text AS violation_type, COUNT(DISTINCT i.name) AS bank_count
        ORDER BY bank_count DESC
        RETURN violation_type, bank_count
      Never apply LIMIT 1 to a GROUP BY — it would return only one group instead of all.
3. RETURN only properties, never full nodes  →  RETURN i.name, p.date  not  RETURN i
4. Use OPTIONAL MATCH for relationships that may not exist (Violation, Regulation).
5. Every MATCH clause must include a label  →  MATCH (i:Institution)
6. Use case-insensitive matching when filtering by name  →  toLower(i.name) CONTAINS toLower("...")
7. For date ranges use string comparison  →  p.date >= "2023-01-01" AND p.date < "2024-01-01"
8. For aggregation across hops use ORDER BY + LIMIT (not COUNT, unless explicitly asked).
9. Whenever a PenaltyAction node (p) is in scope, ALWAYS add BOTH:
     OPTIONAL MATCH (p)-[:FOR_VIOLATION]->(v:Violation)
     OPTIONAL MATCH (p)-[:CITES]->(r:Regulation)
   and include v.text AND r.citation AND p.pr_number in RETURN.
   v.text is the factual description of what went wrong.
   r.citation is the legal basis for the penalty.
   p.pr_number is required for inline citations.
   Include all three regardless of which aspect the question emphasises.
10. Output ONLY the raw Cypher — no explanation, no markdown, no code fences.
11. WITH ALIAS RULE — CRITICAL: every expression in a WITH clause MUST have an AS alias.
    This is required by Neo4j's Cypher parser — unaliased expressions in WITH cause SyntaxError.
    Wrong:  WITH i, p.date, COUNT(v)
    Correct: WITH i, p.date AS p_date, COUNT(v) AS n_violations
    Wrong:  WITH i ORDER BY p.penalty_lakh DESC LIMIT 1
    Correct: WITH i, p.penalty_lakh AS penalty_lakh ORDER BY penalty_lakh DESC LIMIT 1
    Every column in WITH that will be used in subsequent clauses must be named explicitly.
12. COOPERATIVE BANK RULE — always use IN filter for cooperative banks:
    WHERE i.entity_type IN ["Bank-Cooperative","Bank-UCB","Bank-DCCB","Bank-SCB"]
13. COUNT DISAMBIGUATION:
    - Counting distinct institutions → COUNT(DISTINCT i.name) AS institution_count
    - Counting distinct penalty events → COUNT(DISTINCT p.prid) AS penalty_count

EXAMPLE A — top-N lookup (LIMIT 1 to select top entity, then join details):
  MATCH (i:Institution)-[:RECEIVED]->(p:PenaltyAction)
  WHERE p.date >= "2023-01-01" AND p.date < "2024-01-01"
    AND p.penalty_lakh IS NOT NULL
  WITH i, p.penalty_lakh AS penalty_lakh ORDER BY penalty_lakh DESC LIMIT 1
  OPTIONAL MATCH (p)-[:FOR_VIOLATION]->(v:Violation)
  OPTIONAL MATCH (p)-[:CITES]->(r:Regulation)
  RETURN i.name AS entity_name, penalty_lakh, p.date AS p_date, p.pr_number, v.text, r.citation

EXAMPLE B — GROUP BY aggregation (for each X across ALL X, aggregate Y — no LIMIT):
  MATCH (i:Institution)-[:RECEIVED]->(p:PenaltyAction)-[:FOR_VIOLATION]->(v:Violation)
  WHERE i.entity_type IN ["Bank-Private","Bank-Public","Bank-Cooperative","Bank-UCB","Bank-DCCB","Bank-SCB","Bank-RRB","Bank-SFB","Bank-Payment"]
  WITH v.text AS violation_type, COUNT(DISTINCT i.name) AS bank_count
  ORDER BY bank_count DESC
  RETURN violation_type, bank_count

Cypher:"""
