# Evaluation

The evaluation harness is deliberately result-driven. It never prints made-up
scores. Provide JSON produced by an actual pipeline with one result per case.

Required result shape:
```json
{
  "case-001": {
    "answer": "...",
    "evidence": [{"document":"...","page":1,"chunk_id":"...","text":"..."}],
    "claims": [{"claim":"..."}],
    "citations": [{"document":"...","page":1,"chunk_id":"..."}]
  },
  "ablation": [
    {"variant":"A","name":"Basic RAG","aggregate":{...}}
  ]
}
```

The dataset contains small deterministic fixtures for smoke tests. Replace or
extend them with approved legal evaluation material before reporting benchmark
numbers.
