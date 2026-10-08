# Member 3 — Advanced Legal Analysis

Built on current main, including Member 1 retrieval and Member 2 grounding.

## Workflows
- Case / Contract Review: source-preserving facts, evidence, contradictions, and missing information.
- Legal Drafting: grounded source material, explicit missing-information placeholders, Member 2 claim verification, and no invented facts.
- Legal Research: case-context-aware retrieval, authority metadata preservation, source-type distinction, and unverified-authority warnings.
- Contradiction Analysis: CONTRADICTION, POTENTIAL_CONTRADICTION, or NO_CONTRADICTION with both evidence sides.
- Missing Information: NOT_FOUND_IN_PROVIDED_DOCUMENTS; absence is not treated as proof of non-existence.

## Integration
- Member 1: retrieval.hybrid_search.retrieve(query, top_k=5)
- Member 2: grounding.claim_verifier.verify_claims and grounding.citation_validator.validate_citations
- Shared adapter: legal.shared

## Safety
No names, dates, sections, judgments, authorities, citations, or other legal facts are invented. Unknown authority metadata remains unverified.

## Test
pytest tests/test_legal_analysis.py
