from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.reporting import build_report_rows, write_report


def test_report_has_overall_and_task_rows_with_videospy_metrics(tmp_path: Path) -> None:
    records = [
        {"type": "EntityLog", "evaluate": True, "completed": True, "num_rounds": 2, "total_tokens": 10, "elapsed_seconds": 1.0},
        {"type": "EntityLog", "evaluate": False, "completed": False, "num_rounds": 4, "total_tokens": 30, "elapsed_seconds": 3.0},
        {"type": "EventRecall", "evaluate": True, "completed": True, "num_rounds": 1, "total_tokens": 20, "elapsed_seconds": 2.0},
    ]
    rows = build_report_rows(records, agent="WorldMM", mode="test", model="Qwen3.5-4B")
    overall = rows[0]
    assert overall["Task"] == "Overall"
    assert overall["Completed"] == "2/3"
    assert overall["Accuracy (%)"] == 66.7
    assert overall["Avg. rounds"] == 2.3
    entity = next(row for row in rows if row["Task"] == "EntityLog")
    assert entity["Completed"] == "1/2"
    assert entity["Avg. tokens"] == 20.0
    paths = write_report(rows, tmp_path, title="EgoLifeQA Run Report")
    assert paths["json"].is_file()
