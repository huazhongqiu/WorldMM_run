import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval" / "eval.py"


def load_eval_module():
    sys.path.insert(0, str(ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("worldmm_eval_checkpoint", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "src"))


def test_load_latest_results_ignores_partial_tail_and_last_record_wins(tmp_path: Path) -> None:
    module = load_eval_module()
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"ID": "1", "status": "error"}) + "\n"
        + json.dumps({"ID": "1", "status": "success", "response": "B"}) + "\n"
        + '{"ID": "truncated"',
        encoding="utf-8",
    )

    latest = module.load_latest_results(str(path))

    assert latest == {"1": {"ID": "1", "status": "success", "response": "B"}}


def test_pending_rows_retries_errors_and_skips_only_successes() -> None:
    module = load_eval_module()
    rows = [{"ID": "1"}, {"ID": "2"}, {"ID": "3"}]
    latest = {
        "1": {"ID": "1", "status": "success"},
        "2": {"ID": "2", "status": "error"},
    }

    assert module.pending_rows(rows, latest) == [{"ID": "2"}, {"ID": "3"}]


def test_merge_results_preserves_eval_order_and_marks_missing() -> None:
    module = load_eval_module()
    rows = [
        {"ID": "2", "video_id": "v", "question": "q2", "answer": "B"},
        {"ID": "1", "video_id": "v", "question": "q1", "answer": "A"},
    ]
    latest = {"1": {"ID": "1", "status": "success", "response": "A"}}

    merged = module.merge_results(rows, latest)

    assert [row["ID"] for row in merged] == ["2", "1"]
    assert merged[0]["status"] == "missing"
    assert merged[1]["status"] == "success"


def test_completion_counts_distinguishes_success_error_and_missing() -> None:
    module = load_eval_module()
    assert module.completion_counts(
        [
            {"status": "success", "response": "A"},
            {"status": "success", "response": ""},
            {"status": "error"},
            {"status": "missing"},
        ]
    ) == {"success": 1, "error": 2, "missing": 1, "total": 4}
