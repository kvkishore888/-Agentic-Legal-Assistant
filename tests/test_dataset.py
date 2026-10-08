import json
from pathlib import Path


def test_evaluation_dataset_is_traceable():
    path = Path("evaluation/dataset/cases.json")
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert cases
    for case in cases:
        assert case["question"]
        assert case["expected_answer"]
        assert case["expected_sources"]
        assert case["document"]
