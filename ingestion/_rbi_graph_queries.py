"""
ingestion/_rbi_graph_queries.py

All Cypher strings for the RBI enforcement Schema A knowledge graph.
Used exclusively by graph_writer.py.

Schema:
    (Institution)-[:RECEIVED]->(PenaltyAction)-[:FOR_VIOLATION]->(Violation)
                                              -[:CITES]->(Regulation)

Deduplication keys:
    Institution  — name             (MERGE property)
    PenaltyAction — prid            (MERGE property)
    Violation    — text_key         (lowercased + stripped text)
    Regulation   — citation_key     (lowercased + stripped citation)
"""

# ── Constraints ───────────────────────────────────────────────────────────────

CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT rbi_institution_name IF NOT EXISTS "
    "FOR (n:Institution) REQUIRE n.name IS UNIQUE",

    "CREATE CONSTRAINT rbi_penalty_prid IF NOT EXISTS "
    "FOR (n:PenaltyAction) REQUIRE n.prid IS UNIQUE",

    "CREATE CONSTRAINT rbi_violation_key IF NOT EXISTS "
    "FOR (n:Violation) REQUIRE n.text_key IS UNIQUE",

    "CREATE CONSTRAINT rbi_regulation_key IF NOT EXISTS "
    "FOR (n:Regulation) REQUIRE n.citation_key IS UNIQUE",
]

# ── Node merges ───────────────────────────────────────────────────────────────

MERGE_INSTITUTIONS = """
UNWIND $rows AS row
MERGE (i:Institution {name: row.name})
SET i.entity_type = CASE
  WHEN i.entity_type IS NULL OR i.entity_type = 'UNKNOWN'
  THEN row.entity_type
  ELSE i.entity_type
END
"""

MERGE_PENALTY_ACTIONS = """
UNWIND $rows AS row
MERGE (p:PenaltyAction {prid: row.prid})
SET p.date         = row.date,
    p.title        = row.title,
    p.penalty_lakh = row.penalty_lakh,
    p.action_type  = row.action_type,
    p.source_url   = row.source_url,
    p.pr_number    = row.pr_number
"""

# ON CREATE SET preserves whichever text variant was seen first
MERGE_VIOLATIONS = """
UNWIND $rows AS row
MERGE (v:Violation {text_key: row.text_key})
ON CREATE SET v.text = row.text
"""

MERGE_REGULATIONS = """
UNWIND $rows AS row
MERGE (r:Regulation {citation_key: row.citation_key})
ON CREATE SET r.citation = row.citation
"""

# ── Relationship merges ───────────────────────────────────────────────────────

REL_RECEIVED = """
UNWIND $rows AS row
MATCH (i:Institution   {name: row.institution_name})
MATCH (p:PenaltyAction {prid: row.prid})
MERGE (i)-[:RECEIVED]->(p)
"""

REL_FOR_VIOLATION = """
UNWIND $rows AS row
MATCH (p:PenaltyAction {prid:     row.prid})
MATCH (v:Violation     {text_key: row.text_key})
MERGE (p)-[:FOR_VIOLATION]->(v)
"""

REL_CITES = """
UNWIND $rows AS row
MATCH (p:PenaltyAction {prid:         row.prid})
MATCH (r:Regulation    {citation_key: row.citation_key})
MERGE (p)-[:CITES]->(r)
"""

# ── Density check (CLAUDE.md §6) ─────────────────────────────────────────────

DENSITY_CHECK = """
MATCH (n)
RETURN labels(n)[0] AS type, count(n) AS count,
       avg(COUNT { (n)-->() }) AS avg_out_degree
ORDER BY count DESC
LIMIT 20
"""
