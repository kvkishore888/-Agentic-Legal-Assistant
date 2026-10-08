"""Deterministic evaluation harness with explicit ablation support."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any,Callable
from .metrics import aggregate,evaluate_result
ABLATIONS={"A":"basic_rag","B":"hybrid_retrieval","C":"hybrid_plus_reranking","D":"claim_verification","E":"citation_verification","F":"full_system"}
def load_dataset(path="evaluation/dataset/cases.json"): return json.loads(Path(path).read_text(encoding="utf-8"))
def evaluate_pipeline(pipeline:Callable[[dict[str,Any]],dict[str,Any]],dataset_path="evaluation/dataset/cases.json"):
    rows=[]
    for case in load_dataset(dataset_path): rows.append(evaluate_result(case,pipeline(case)))
    return {"cases":rows,"aggregate":aggregate(rows)}
def validate_result_schema(result):
    required={"answer","evidence","claims","citations","confidence","missing_information"}
    missing=required-set(result)
    if missing: raise ValueError(f"Evaluation result missing fields: {sorted(missing)}")
    return True
def run_ablation(pipelines,dataset_path="evaluation/dataset/cases.json"):
    out={}
    for key,name in ABLATIONS.items():
        if name not in pipelines: continue
        out[key]=evaluate_pipeline(pipelines[name],dataset_path)
    return out
