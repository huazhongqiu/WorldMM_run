import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("B", "B"),
        ("The evidence supports the drum kit.\n\nFinal answer: B.", "B"),
        ("Reasoning text.\n\nD", "D"),
        ("Based on the frames, this cannot be determined.", None),
    ],
)
def test_extract_final_choice_letter_uses_explicit_or_terminal_choice(
    response: str, expected: str | None
) -> None:
    module = load_eval_module()

    assert module.extract_final_choice_letter(
        response, {"A": "first", "B": "second", "C": "third", "D": "fourth"}
    ) == expected


def test_evaluate_prediction_uses_terminal_choice_in_reasoning_text() -> None:
    module = load_eval_module()

    assert module.evaluate_prediction("Reasoning text.\n\nA", "A", {"A": "first"})


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
        "1": {"ID": "1", "status": "success", "response": "A"},
        "2": {"ID": "2", "status": "error"},
    }

    assert module.pending_rows(rows, latest) == [{"ID": "2"}, {"ID": "3"}]


def test_generation_failure_sentinel_is_not_successful() -> None:
    module = load_eval_module()
    result = {
        "ID": "1",
        "status": "success",
        "response": "Unable to generate answer",
    }

    assert not module.is_successful_result(result)
    assert module.answer_status(result["response"]) == "generation_error"


def test_required_openie_missing_or_malformed_fails_closed(tmp_path: Path) -> None:
    module = load_eval_module()

    class FakeMemory:
        def load_episodic_openie(self, path):
            raise ValueError("malformed")

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        module.load_required_episodic_openie(FakeMemory(), str(missing))

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        module.load_required_episodic_openie(FakeMemory(), str(malformed))


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
