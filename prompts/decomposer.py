"""
prompts/decomposer.py

Question decomposition prompt for the Stage 3 LangGraph decomposer node.

Export:
  PROMPT -- format with {max_q} and {question}
"""

PROMPT = """\
Break the following question into at most {max_q} atomic sub-questions that \
can each be answered independently from a single source.
If the question is already simple and self-contained, return it unchanged as item 1.

Rules:
- Output ONLY a numbered list (1. ... 2. ...), no other text
- Each item must be a complete standalone question
- Maximum {max_q} items

Question: {question}"""
