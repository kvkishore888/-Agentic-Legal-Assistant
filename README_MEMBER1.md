# Member 1 — Document Ingestion & Retrieval

This branch provides page-preserving legal PDF ingestion, chunking, embeddings, persistent Chroma storage, keyword search, hybrid retrieval, and optional reranking.

## Shared interface

`retrieve(query, top_k=5)` is exposed by `retrieval.hybrid_search`.

Configure it once:

```python
from retrieval.hybrid_search import HybridRetriever, configure
retriever=HybridRetriever()
retriever.index(chunks)
configure(retriever)
```

Every result preserves `document`, `page`, `chunk_id`, `text`, and `score`.

## OCR

OCR is optional and requires the Tesseract system executable in addition to the Python packages. The ingestion layer exposes `needs_ocr` and `ocr_page` without silently fabricating text.

## Tests

Run `pytest`.
