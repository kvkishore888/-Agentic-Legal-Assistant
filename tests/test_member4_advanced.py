from evaluation.harness import evaluate_pipeline,validate_result_schema,run_ablation
def result(case):
    return {"answer":case["expected_answer"],"evidence":case["expected_sources"],"claims":[{"claim_text":case["expected_answer"],"status":"SUPPORTED"}],"citations":case["expected_sources"],"confidence":"HIGH","missing_information":[]}
def test_schema(): assert validate_result_schema(result({"expected_answer":"x","expected_sources":[],"id":"x"}))
def test_evaluation(): 
    out=evaluate_pipeline(result,"evaluation/dataset/cases.json"); assert out["aggregate"]["fabricated_claim_rate"]==0.0
def test_ablation_only_runs_available():
    out=run_ablation({"full_system":result}); assert "F" in out and out["F"]["cases"]
