# Evaluation dataset

The evaluation harness is wired into the application, but **real judging data must be sourced from public legal documents**. This directory intentionally does not contain fabricated case answers or invented source references.

For the hackathon, add JSON cases with:
- `id`
- `query`
- `expected_answer`
- `expected_sources`: exact `document`, `page`, and `chunk_id`
- `workflow`

The source identifiers must come from the actual PDFs after ingestion. Do not copy placeholder IDs into the held-out evaluation set.

Recommended held-out split:
- 10 grounded RAG questions
- 8 case/contract review questions
- 5 contradiction questions
- 5 missing-information questions
- 5 drafting questions
- 5 legal-research questions
- 7 adversarial / unsupported-claim questions

Run baseline and full-system pipelines on exactly the same queries, then report groundedness, retrieval recall@k, citation correctness, fabricated-claim rate, usefulness, and an ablation comparison. Never publish fabricated evaluation numbers.
