"""
prompts/hyde.py

HyDE (Hypothetical Document Embedding) prompt for Stage 2 Advanced RAG.

The model generates a short passage in the style of an RBI enforcement press
release, which is then embedded instead of the raw query.  This closes the
vocabulary gap between user queries and corpus text.

Export:
  TEMPLATE -- format with {question} to produce the HyDE prompt
"""

TEMPLATE = """\
Write a short factual passage that would appear in an RBI enforcement press \
release directly addressing the following question. Use formal regulatory \
language typical of Reserve Bank of India press releases (e.g. "The Reserve \
Bank of India has imposed a monetary penalty of...", "...in exercise of the \
powers vested in RBI under Section..."). Write as a source document excerpt, \
not as a conversational answer.

Question: {question}
Passage:"""
