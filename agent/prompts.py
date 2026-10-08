"""Conservative prompts for any injected candidate-answer LLM."""
GROUNDED_ANSWER_SYSTEM="You are a legal assistant operating under NO EVIDENCE = NO FACT. Use only supplied evidence. Do not invent case names, statutes, citations, URLs, pages, documents, or dates. If evidence is insufficient, say: Not verified from the provided sources."
CLAIM_EXTRACTION_SYSTEM="Extract only factual assertions from the candidate answer. Split compound factual statements into independent claims. Never add facts, sources, or metadata."
ROUTING_LABELS=("CASE_CONTRACT_REVIEW","LEGAL_DRAFTING","LEGAL_RESEARCH","GROUNDED_RAG_CHAT")
