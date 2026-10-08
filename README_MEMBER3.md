# Member 3 — Legal Analysis Workflows

Branch: member-3-legal-analysis

## Public APIs

- legal.case_review.review_case(query, top_k=5): source-preserving case/contract review.
- legal.contradiction.find_contradictions(evidence): conservative contradiction classification.
- legal.missing_info.detect_missing_information(hits, required): reports only NOT_FOUND_IN_PROVIDED_DOCUMENTS.
- legal.drafting.draft_document(document_type, query, ...): grounded structured drafting with placeholders.
- legal.research.research(query, case=None, top_k=5): retrieval-connected authority research.

## Shared interfaces

Member 1: retrieval.hybrid_search.retrieve(query, top_k=5).

Member 2: claim verification and citation validation are consumed through legal.shared.GroundingService; the adapter can auto-discover compatible grounding.* modules or be explicitly configured with configure_grounding(...).

## Safety

No legal fact, party, date, section, judgment, or citation is invented. Retrieval metadata is retained on findings. Missing material is reported as not found in provided documents, never as proof that it does not exist. External authorities are not labeled authoritative unless citation validation succeeds.

## Tests

pytest tests/test_legal_analysis.py
