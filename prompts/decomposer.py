"""
prompts/decomposer.py

Question decomposition prompt for the Stage 5 LangGraph decomposer node.

Stage 5 change: the decomposer now outputs JSON instead of a numbered list.
This enables the pipeline to detect:
  - dependent sub-questions (template variables like {q0_answer})
  - whether all hops are graph-only (cypher_sufficient=True → single Cypher)
  - whether serial execution is required (execution_mode="serial")

Export:
  PROMPT -- format with {max_q} and {question}
"""

PROMPT = """\
You are a query planner for an RBI enforcement database that has:
  - A Neo4j knowledge graph (Institution, PenaltyAction, Violation, Regulation nodes)
  - A ChromaDB vector store of RBI press releases

Break the following question into at most {max_q} atomic sub-questions.
Then classify whether they can run in parallel or must run in serial order.

OUTPUT FORMAT — respond with ONLY valid JSON, no explanation, no markdown fences:
{{
  "sub_questions": [
    {{"id": 0, "q": "<sub-question text>", "depends_on": null}},
    {{"id": 1, "q": "<sub-question using {{q0_answer}} if it needs hop-0 result>", "depends_on": 0}}
  ],
  "execution_mode": "parallel" | "serial",
  "cypher_sufficient": true | false
}}

RULES:
1. If the question is already simple and self-contained, return it unchanged as id=0 with depends_on=null.
2. execution_mode is "serial" when ANY sub-question depends on a prior sub-question's answer.
   Otherwise "parallel".
3. cypher_sufficient is true ONLY when EVERY sub-question can be answered purely from the
   Neo4j graph (structured data: names, dates, penalty amounts, violation types, regulation citations).
   Set false if ANY sub-question requires reasoning over press release text.
4. For dependent sub-questions, embed the template variable {{q0_answer}}, {{q1_answer}}, etc.
   in the question text wherever the prior hop's answer is needed.
   Example: "What violations did {{q0_answer}} commit?" depends_on: 0
5. Maximum {max_q} sub-questions total.
6. Each sub-question must be a complete, standalone question.
7. depends_on must reference an id with a LOWER number (no forward references).

EXAMPLES:

Simple factual question → parallel, single hop:
Input:  "What penalty did Axis Bank receive in 2023?"
Output: {{"sub_questions": [{{"id": 0, "q": "What penalty did Axis Bank receive in 2023?", "depends_on": null}}], "execution_mode": "parallel", "cypher_sufficient": true}}

Multi-hop with dependency → serial, hybrid:
Input:  "For the bank with the highest penalty in 2023, what were the specific violations?"
Output: {{
  "sub_questions": [
    {{"id": 0, "q": "Which bank received the highest penalty in 2023?", "depends_on": null}},
    {{"id": 1, "q": "What were the specific violations committed by {{q0_answer}}?", "depends_on": 0}}
  ],
  "execution_mode": "serial",
  "cypher_sufficient": false
}}

Multi-hop fully on graph → serial, cypher_sufficient:
Input:  "In 2023, which bank had the most penalties and what was the total amount?"
Output: {{
  "sub_questions": [
    {{"id": 0, "q": "Which bank received the most penalties in 2023?", "depends_on": null}},
    {{"id": 1, "q": "What was the total penalty amount imposed on {{q0_answer}} in 2023?", "depends_on": 0}}
  ],
  "execution_mode": "serial",
  "cypher_sufficient": true
}}

Question: {question}"""
