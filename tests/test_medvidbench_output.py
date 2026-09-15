import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    sys.path.insert(0, str(ROOT / "src"))
    try:
        path = ROOT / "eval" / "eval_medvidbench.py"
        spec = importlib.util.spec_from_file_location("medvidbench_output_runner", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "src"))


@pytest.mark.parametrize(
    ("qa_type", "response", "expected"),
    [
        (
            "tal",
            "The events occur at 1.1 - 3.0 seconds and 5.0-7.0 seconds.",
            "1.1-3.0, 5.0-7.0 seconds.",
        ),
        (
            "stg",
            "Answer: 2.0 seconds: [1, 2, 3, 4]\n4 seconds: [5.5, 6, 7, 8]",
            "2.0 seconds: [1, 2, 3, 4]\n4 seconds: [5.5, 6, 7, 8]",
        ),
        (
            "dense_captioning_gpt",
            "Sure.\n1.0 - 2.5 seconds: dissecting: exposes the cystic duct.",
            "1.0-2.5 seconds: dissecting: exposes the cystic duct.",
        ),
        (
            "skill_assessment",
            "Respect for tissue: 4 / 5; Suture/needle handling: 3/5; Time and motion: 2/5; Flow of operation: 5/5; Overall performance: 4/5; Quality of final product: 3/5",
            "Respect for tissue: 4/5, Suture/needle handling: 3/5, Time and motion: 2/5, Flow of operation: 5/5, Overall performance: 4/5, Quality of final product: 3/5",
        ),
        (
            "cvs_assessment",
            "Assessment - Two structures: 2; Cystic plate: 1; Hepatocystic triangle: 0.",
            "Two structures: 2, Cystic plate: 1, Hepatocystic triangle: 0",
        ),
        ("next_action", "Next action: 'Grasp the cystic duct.'", "Grasp the cystic duct"),
        ("video_summary_gpt", "Answer:\nThe surgeon clips and divides the duct.", "The surgeon clips and divides the duct."),
    ],
)
def test_normalize_prediction_repairs_common_contract_violations(qa_type, response, expected):
    assert _runner().normalize_prediction(qa_type, response) == expected


@pytest.mark.parametrize(
    ("qa_type", "response"),
    [
        ("tal", "The event occurs near the beginning."),
        ("stg", "2.0 seconds: no bounding box was found."),
        ("dense_captioning_gpt", "The surgeon performs a dissection."),
        ("skill_assessment", "Overall performance: 4/5"),
        ("cvs_assessment", "Two structures: 3, Cystic plate: 1, Hepatocystic triangle: 0"),
    ],
)
def test_normalize_prediction_rejects_unrepairable_contract_violations(qa_type, response):
    assert _runner().normalize_prediction(qa_type, response) is None


def test_runner_retries_success_record_with_unrepairable_format():
    module = _runner()
    rows = [{"ID": "1", "video_id": "segment-a"}]
    groups = {"segment-a": rows}
    latest = {
        "1": {
            "status": "success",
            "qa_type": "tal",
            "prediction": "The event occurs near the beginning.",
        }
    }

    assert module.completion_counts(rows, latest) == {
        "success": 0,
        "error": 1,
        "missing": 0,
        "total": 1,
    }
    assert module.pending_groups(groups, latest) == groups
