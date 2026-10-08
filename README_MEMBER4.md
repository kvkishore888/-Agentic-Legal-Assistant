# Member 4 — Advanced Frontend & Evaluation

Built directly on current `main`, integrating Members 1–3.

## Frontend
The Streamlit application is a **single integrated app** with four workflows:
1. Grounded RAG Chat
2. Case / Contract Review
3. Legal Drafting
4. Legal Research

The UI uploads PDFs, runs the existing ingestion/retrieval pipeline, dispatches to the actual legal workflow, and surfaces evidence, claims, citations, contradictions, missing information, warnings and confidence.

## Evaluation
The evaluation layer measures:
- retrieval quality / recall@k
- groundedness
- citation correctness
- fabricated claim rate
- answer usefulness

`evaluation/harness.py` supports reproducible pipeline evaluation and explicit A–F ablations. It never invents scores: a pipeline must supply its actual result observations.

## Run
`streamlit run app/streamlit_app.py`

## Evaluate
Provide actual pipeline results or inject a callable pipeline into `evaluation.harness.evaluate_pipeline`. Do not report benchmark numbers without running the pipeline.
