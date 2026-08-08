"""
prompts/

All LLM prompt strings in one place.
Import by name; never hard-code prompt text in pipeline modules.

  prompts.synthesis   — SYSTEM, AGENTIC_SYSTEM, USER_TEMPLATE (answer generation)
  prompts.hyde        — TEMPLATE (hypothetical document generation)
  prompts.cypher      — SCHEMA, PROMPT (NL → Cypher generation)
  prompts.decomposer  — PROMPT (question decomposition)
  prompts.extraction  — PROMPT_STATIC, build_prompt (entity/violation extraction)
"""
