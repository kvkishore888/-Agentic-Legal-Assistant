"""Run evaluation cases and ablations using real pipeline outputs only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import (
    aggregate, answer_usefulness, citation_correctness, fabricated_claim_rate,
    groundedness, retrieval_recall_at_k,
)


ABLATIONS = [
    ("A", "Basic RAG"),
    ("B", "Basic RAG + Hybrid Retrieval"),
    ("C", "+ Reranking"),
    ("D", "+ Claim Verification"),
    ("E", "+ Citation Verification"),
    ("F", "Full System"),
]


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    evidence = result.get("evidence") or []
    claims = result.get("claims") or []
    citations = result.get("citations") or []
    return {
        "id": case["id"],
        "groundedness": groundedness(claims, evidence),
        "retrieval_quality": retrieval_recall_at_k(evidence, case.get("expected_sources", [])),
        "citation_correctness": citation_correctness(citations, case.get("expected_sources", [])),
        "fabricated_claim_rate": fabricated_claim_rate(claims, evidence),
        "answer_usefulness": answer_usefulness(result.get("answer", ""), case.get("expected_answer", "")),
    }


def run(pipeline: Any, dataset_path: str) -> dict[str, Any]:
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = pipeline(case["question"], case)
        rows.append(evaluate_case(case, result))
    return {"cases": rows, "aggregate": aggregate(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run actual Agentic Legal Assistant evaluation.")
    parser.add_argument("--results", help="JSON file produced by an external pipeline.")
    parser.add_argument("--dataset", default="evaluation/dataset/cases.json")
    args = parser.parse_args()

    if not args.results:
        raise SystemExit("No pipeline results supplied. Refusing to invent evaluation results.")
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = [evaluate_case(case, payload[case["id"]]) for case in
            json.loads(Path(args.dataset).read_text(encoding="utf-8"))]
    output = {"cases": rows, "aggregate": aggregate(rows), "ablation": payload.get("ablation", [])}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
