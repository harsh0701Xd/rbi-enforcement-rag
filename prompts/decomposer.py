"""
prompts/decomposer.py

Question decomposition prompt for the Stage 6 LangGraph decomposer node.

Stage 6 changes:
  - execution_mode removed (always "serial" in Stage 6 — all paths through hop_executor)
  - fan_out added: true when Q2 must iterate over ALL Q1 result rows
  - placeholder syntax updated: {q0.column_name} instead of {q0_answer}
    e.g. {q0.entity_name}, {q0.year}, {q0.penalty_lakh}, {q0.state}

Export:
  PROMPT -- format with {max_q} and {question}
"""

PROMPT = """\
You are a query planner for an RBI enforcement database that has:
  - A Neo4j knowledge graph (Institution, PenaltyAction, Violation, Regulation nodes)
  - A ChromaDB vector store of RBI press releases

Break the following question into at most {max_q} atomic sub-questions.

OUTPUT FORMAT — respond with ONLY valid JSON, no explanation, no markdown fences:
{{
  "sub_questions": [
    {{"id": 0, "q": "<sub-question text>", "depends_on": null}},
    {{"id": 1, "q": "<sub-question using {{q0.field_name}} if it needs hop-0 result>", "depends_on": 0}}
  ],
  "cypher_sufficient": true | false,
  "fan_out": true | false,
  "routes": ["graph" | "vector" | "both", ...]
}}

ROUTING RULES — one entry in "routes" per sub-question, in the same order:
  "graph"  : sub-question is answerable from structured graph data — penalty amounts,
             dates, counts, aggregations, violation types, regulation citations,
             institution names, ranked lookups, yes/no existence checks.
             DEFAULT — use unless "vector" or "both" conditions below apply.
             CRITICAL — violations and regulations are ALWAYS graph-answerable,
             regardless of phrasing. Route to "graph" for:
               "what violations were cited"       → v.text on Violation nodes
               "what regulations were breached"   → r.citation on Regulation nodes
               "what did RBI cite/penalise for"   → v.text + r.citation
             Do NOT route these to "vector" just because the word "press release"
             appears in the question.
  "vector" : sub-question explicitly asks for unstructured press release NARRATIVE that
             is NOT captured in structured fields. Use ONLY for:
               "what did the press release say about [conduct/reasons/background]"
               "how did RBI describe/characterise the misconduct"
               "what language/tone did RBI use in the announcement"
               "what remediation or corrective steps did RBI mention"
               "what background context or inspection details did RBI provide"
             Do NOT use for violation types or regulation citations — those are in the graph.
  "both"   : sub-question names a specific institution AND would benefit from BOTH
             structured graph data AND press release narrative context together.

RULES:
1. If the question is already simple and self-contained, return it as id=0 with depends_on=null.
2. cypher_sufficient is true when EVERY sub-question can be answered purely from the Neo4j graph
   (structured data: names, dates, amounts, counts, violation types, regulation citations).
   This includes aggregation patterns such as "for each X, count/sum/list Y" — the Cypher
   generator can handle these in a single WITH-chained query. Set false only when a sub-question
   requires reasoning over unstructured press release narrative text.
3. fan_out is true ONLY when Q2 requires per-entity vector retrieval — i.e., when the answer
   needs semantic search over press release text for each Q1 result individually.
   Do NOT use fan_out=true for aggregation patterns ("for each X, count Y", "for each X, list Y
   violations") — those are graph-answerable; use cypher_sufficient=true instead.
4. For dependent sub-questions, embed {{q0.field_name}} (e.g. {{q0.entity_name}}, {{q0.year}},
   {{q0.penalty_lakh}}) in the question text where the prior hop's answer is needed.
   Use the exact column name that the Cypher query will return for hop 0.
   Standard column names: entity_name (institution name), year (4-digit year),
   penalty_lakh (amount in lakh), state (institution's state/UT).
   Example: "What violations did {{q0.entity_name}} commit?" depends_on: 0
5. Maximum {max_q} sub-questions total.
6. Each sub-question must be a complete, standalone question.
7. depends_on must reference an id with a LOWER number (no forward references).
8. DATE RANGES — always expand bare year references to explicit date ranges in sub-question text.
   Two cases:
   a) ANNUAL queries (totals, counts, comparisons by year, "in YYYY", "during YYYY"):
      Default to RBI financial year (April 1 – March 31) unless the user says "calendar year".
        "in 2021"            → "between April 1, 2021 and March 31, 2022 (FY2021-22)"
        "during 2023"        → "between April 1, 2023 and March 31, 2024 (FY2023-24)"
        "calendar year 2022" → "between January 1, 2022 and December 31, 2022"
   b) INTRA-YEAR queries (month, quarter, or specific period WITHIN a year):
      Use calendar year (January 1 – December 31) — months and quarters are calendar concepts.
        "which month of 2023"  → "which month between January 1, 2023 and December 31, 2023"
        "Q3 of 2022"           → "between July 1, 2022 and September 30, 2022"
   This applies to ALL sub-questions, including Q1 and dependent hops.

EXAMPLES:

Simple factual question → single hop, graph only:
Input:  "What penalty did Axis Bank receive in 2023?"
Output: {{"sub_questions": [{{"id": 0, "q": "What penalty did Axis Bank receive between April 1, 2023 and March 31, 2024 (FY2023-24)?", "depends_on": null}}], "cypher_sufficient": true, "fan_out": false, "routes": ["graph"]}}

Single hop — pure press release narrative (vector only):
Input:  "What did RBI say about HDFC Bank's conduct in its penalty press release?"
Output: {{"sub_questions": [{{"id": 0, "q": "What did the RBI press release say about HDFC Bank's conduct?", "depends_on": null}}], "cypher_sufficient": false, "fan_out": false, "routes": ["vector"]}}

Multi-hop — Q1 graph lookup, Q2 press release narrative:
Input:  "Which bank received the single largest penalty ever, and what did the RBI press release say about the reasons?"
Output: {{
  "sub_questions": [
    {{"id": 0, "q": "Which bank received the single largest penalty ever recorded?", "depends_on": null}},
    {{"id": 1, "q": "What did the RBI press release say about the reasons for the penalty imposed on {{q0.entity_name}}?", "depends_on": 0}}
  ],
  "cypher_sufficient": false,
  "fan_out": false,
  "routes": ["graph", "vector"]
}}

Multi-hop — Q1 graph, Q2 graph+narrative (both):
Input:  "For the bank with the highest penalty in 2023, what were the specific violations and what did the press release say?"
Output: {{
  "sub_questions": [
    {{"id": 0, "q": "Which bank received the highest penalty between April 1, 2023 and March 31, 2024 (FY2023-24)?", "depends_on": null}},
    {{"id": 1, "q": "What violations did {{q0.entity_name}} commit and what did the RBI press release say about their conduct?", "depends_on": 0}}
  ],
  "cypher_sufficient": false,
  "fan_out": false,
  "routes": ["graph", "both"]
}}

Graph aggregation "for each X, count/list Y" → cypher_sufficient (NOT fan_out):
Input:  "For each cooperative bank penalised in 2023, how many violations did they commit?"
Output: {{
  "sub_questions": [
    {{"id": 0, "q": "Which cooperative banks were penalised between April 1, 2023 and March 31, 2024 (FY2023-24)?", "depends_on": null}},
    {{"id": 1, "q": "How many violations did {{q0.entity_name}} commit?", "depends_on": 0}}
  ],
  "cypher_sufficient": true,
  "fan_out": false,
  "routes": ["graph", "graph"]
}}

"For each X across ALL X" → single sub-question (GROUP BY, not iteration):
NOTE: When "for each X" means ALL instances of X in the database (not a filtered subset),
it is a single GROUP BY aggregation — do NOT decompose into Q1+Q2.
Input:  "For each violation type, how many banks committed it?"
Output: {{"sub_questions": [{{"id": 0, "q": "For each violation type, how many distinct banks committed it?", "depends_on": null}}], "cypher_sufficient": true, "fan_out": false, "routes": ["graph"]}}

Input:  "For each year, what was the total penalty amount imposed?"
Output: {{"sub_questions": [{{"id": 0, "q": "For each year, what was the total penalty amount imposed across all banks?", "depends_on": null}}], "cypher_sufficient": true, "fan_out": false, "routes": ["graph"]}}

Multi-hop with year dependency → cypher_sufficient, all graph:
Input:  "What was the largest penalty Axis Bank received, and how does that compare to HDFC Bank in the same year?"
Output: {{
  "sub_questions": [
    {{"id": 0, "q": "What was the largest penalty Axis Bank received and in what financial year?", "depends_on": null}},
    {{"id": 1, "q": "What penalties did HDFC Bank receive in {{q0.year}}?", "depends_on": 0}}
  ],
  "cypher_sufficient": true,
  "fan_out": false,
  "routes": ["graph", "graph"]
}}

Fan-out — per-entity press release text needed (vector required per entity):
Input:  "For each bank penalised in 2023, what remediation steps did the RBI press release mention?"
Output: {{
  "sub_questions": [
    {{"id": 0, "q": "Which banks were penalised between April 1, 2023 and March 31, 2024 (FY2023-24)?", "depends_on": null}},
    {{"id": 1, "q": "What remediation steps did the RBI press release mention for {{q0.entity_name}}?", "depends_on": 0}}
  ],
  "cypher_sufficient": false,
  "fan_out": true,
  "routes": ["graph", "vector"]
}}

State-level aggregation → single sub-question (GROUP BY i.state, graph only):
Input:  "For each state, how many cooperative banks were penalised in 2023?"
Output: {{"sub_questions": [{{"id": 0, "q": "For each state, how many cooperative banks were penalised between April 1, 2023 and March 31, 2024 (FY2023-24)?", "depends_on": null}}], "cypher_sufficient": true, "fan_out": false, "routes": ["graph"]}}

Multi-hop with state dependency:
Input:  "Which state had the most penalised cooperative banks in 2023, and what violations did they commit?"
Output: {{
  "sub_questions": [
    {{"id": 0, "q": "Which state had the most cooperative banks penalised between April 1, 2023 and March 31, 2024 (FY2023-24)?", "depends_on": null}},
    {{"id": 1, "q": "What violations did cooperative banks in {{q0.state}} commit between April 1, 2023 and March 31, 2024 (FY2023-24)?", "depends_on": 0}}
  ],
  "cypher_sufficient": true,
  "fan_out": false,
  "routes": ["graph", "graph"]
}}

Question: {question}"""
