import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = ROOT / "data" / "MedVidBench" / "utils"


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "src"))


def _prepare():
    return _load_module("medvidbench_prepare", UTILS_DIR / "prepare_medvidbench.py")


def _evaluate():
    return _load_module("medvidbench_evaluate", UTILS_DIR / "evaluate_medvidbench.py")


# ---------------------------------------------------------------------------
# prepare_medvidbench.py
# ---------------------------------------------------------------------------

ALL_QA_TYPES = [
    "tal",
    "stg",
    "dense_captioning_gpt",
    "dense_captioning_gemini",
    "video_summary_gpt",
    "video_summary_gemini",
    "region_caption_gpt",
    "region_caption_gemini",
    "next_action",
    "skill_assessment",
    "cvs_assessment",
]


def test_output_requirement_covers_every_qa_type():
    module = _prepare()
    for qa_type in ALL_QA_TYPES:
        requirement = module.output_requirement(qa_type)
        assert requirement, f"missing output contract for {qa_type}"
        assert requirement.startswith("Return")


def test_build_question_appends_videospy_contract_verbatim():
    module = _prepare()
    sample = {
        "qa_type": "tal",
        "conversations": [
            {"from": "human", "value": "<video>\nWhen does the event happen?"},
            {"from": "gpt", "value": "1.0-3.0 seconds."},
        ],
    }
    question = module.build_question(sample)
    assert question.startswith("When does the event happen?")
    assert question.endswith(
        'Output requirement: Return only the time span(s), for example: "1.1-3.0 seconds." '
        'For multiple spans, use: "1.1-3.0, 5.0-7.0 seconds."'
    )


def test_segment_key_is_safe_unique_and_stable():
    module = _prepare()
    raw_id = "1.0$014014_part0$184$324"
    base = module.segment_key(raw_id)
    assert "&&" not in base and "$" not in base and "/" not in base
    assert base == module.segment_key(raw_id)
    rc_one = module.segment_key(raw_id, '["[1, 2, 3, 4]"]')
    rc_two = module.segment_key(raw_id, '["[5, 6, 7, 8]"]')
    assert rc_one != rc_two != base
    assert base in (rc_one, rc_two) or base != rc_one


def test_build_row_schema_matches_worldmm_qa_format():
    module = _prepare()
    item = {
        "id": "1.0$014014_part0$184$324",
        "qa_type": "dense_captioning_gpt",
        "dataset_name": "CoPESD",
        "is_RC": False,
        "metadata": {"fps": 1.0},
        "video": ["f1", "f2", "f3", "f4"],
        "conversations": [
            {"from": "human", "value": "<video>\nDescribe the events."},
            {"from": "gpt", "value": "0.0-4.0 seconds: resection: labels the lesion."},
        ],
    }
    row = module.build_row(42, item, "key-abc", 4.0)
    assert row["ID"] == "42"
    assert row["video_id"] == "key-abc"
    assert row["query_time"] == {"date": "DAY1", "time": "23595999"}
    assert row["type"] == "dense_captioning_gpt"
    assert row["task"] == "dvc"
    assert row["duration"] == "short"
    assert row["domain"] == "CoPESD"
    assert "Output requirement:" in row["question"]
    assert row["answer"] == "0.0-4.0 seconds: resection: labels the lesion."
    assert row["raw_id"] == item["id"]
    assert not any(row.get(f"choice_{c}") for c in "abcd")


def test_duration_buckets():
    module = _prepare()
    assert module.duration_bucket(29.9) == "short"
    assert module.duration_bucket(30.0) == "medium"
    assert module.duration_bucket(300.0) == "medium"
    assert module.duration_bucket(300.1) == "long"


# ---------------------------------------------------------------------------
# evaluate_medvidbench.py
# ---------------------------------------------------------------------------

OFFICIAL_OUTPUT = """
some progress lines
LEADERBOARD METRICS SUMMARY
  mIoU@0.3: 0.4120
  mIoU@0.5: 0.2330
  mean_iou: 0.3100
  accuracy: 0.5500
  temporal_f1: 0.1200
  caption_score: 3.1500
  aspect_balanced_accuracy: 0.4000
  component_balanced_accuracy: 0.3333
VS - Overall Evaluation
  score: 3.4000
RC - Overall Evaluation
  score: 2.9000
END LEADERBOARD METRICS SUMMARY
trailing lines
"""


def test_parse_official_metrics():
    module = _evaluate()
    metrics = module.parse_official_metrics(OFFICIAL_OUTPUT)
    assert metrics["tag_miou_03"] == 0.4120
    assert metrics["tag_miou_05"] == 0.2330
    assert metrics["stg_miou"] == 0.31
    assert metrics["nap_acc"] == 0.55
    assert metrics["dvc_f1"] == 0.12
    assert metrics["dvc_llm"] == 3.15
    assert metrics["sa_acc"] == 0.40
    assert metrics["cvs_acc"] == 0.3333
    assert metrics["vs_llm"] == 3.40
    assert metrics["rc_llm"] == 2.90


def test_select_ground_truth_and_alignment():
    module = _evaluate()
    gt = [{"id": f"vid&&{i}&&{i + 5}&&1.0", "qa_type": "tal"} for i in range(10)]
    selected = module.select_ground_truth(gt, ["3", "7"])
    assert [row["id"] for row in selected] == [gt[3]["id"], gt[7]["id"]]
    submission = [
        {"id": gt[3]["id"], "qa_type": "tal", "prediction": "1.0-2.0 seconds."},
        {"id": gt[7]["id"], "qa_type": "tal", "prediction": ""},
    ]
    module.validate_alignment(submission, selected)
    bad = [{"id": "other", "qa_type": "tal", "prediction": ""}]
    try:
        module.validate_alignment(bad, selected)
        raise AssertionError("expected alignment failure")
    except ValueError:
        pass


def test_nap_exact_accuracy():
    module = _evaluate()
    submission = [
        {"qa_type": "next_action", "prediction": " Grasping "},
        {"qa_type": "next_action", "prediction": "Cutting"},
        {"qa_type": "tal", "prediction": "ignored"},
    ]
    gt = [
        {"conversations": [{"from": "gpt", "value": "grasping"}]},
        {"conversations": [{"from": "gpt", "value": "suturing"}]},
        {"conversations": [{"from": "gpt", "value": "ignored"}]},
    ]
    assert module.compute_nap_exact_accuracy(submission, gt) == 0.5
