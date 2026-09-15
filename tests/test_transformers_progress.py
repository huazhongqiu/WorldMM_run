from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_runners_disable_transformers_checkpoint_progress() -> None:
    for path in (ROOT / "eval" / "eval.py", ROOT / "eval" / "eval_medvidbench.py"):
        source = path.read_text(encoding="utf-8")

        assert "transformers_logging.disable_progress_bar()" in source
