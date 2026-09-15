import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval" / "validate_precomputed.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_precomputed", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_tree(tmp_path: Path, *, model: str = "Qwen3.5-4B") -> tuple[Path, Path]:
    root = tmp_path / "prepared"
    rows = [
        {"ID": "1", "video_id": "video-a"},
        {"ID": "2", "video_id": "video-a"},
        {"ID": "3", "video_id": "video-b"},
    ]
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps(rows), encoding="utf-8")
    for video_id in ("video-a", "video-b"):
        caption = root / "caption" / video_id
        caption.mkdir(parents=True)
        for granularity in ("10sec", "30sec", "3min", "10min"):
            (caption / f"{granularity}.json").write_text("[]", encoding="utf-8")
        episodic = root / "episodic_memory" / video_id
        episodic.mkdir(parents=True)
        (episodic / f"openie_results_{model}.json").write_text("[]", encoding="utf-8")
        (episodic / f"episodic_triple_results_{model}.json").write_text("[]", encoding="utf-8")
        semantic = root / "semantic_memory" / video_id
        semantic.mkdir(parents=True)
        (semantic / f"semantic_consolidation_results_{model}.json").write_text("[]", encoding="utf-8")
        visual = root / "visual_memory" / video_id
        visual.mkdir(parents=True)
        (visual / "visual_embeddings.pkl").write_bytes(b"prepared")
    return eval_json, root


def test_validate_precomputed_accepts_complete_tree(tmp_path: Path) -> None:
    module = load_validator()
    eval_json, root = build_tree(tmp_path)

    summary = module.validate_precomputed(eval_json, root, "Qwen3.5-4B")

    assert summary.question_count == 3
    assert summary.unit_count == 2
    assert summary.missing == ()


def test_validate_precomputed_reports_missing_paths_in_stable_order(tmp_path: Path) -> None:
    module = load_validator()
    eval_json, root = build_tree(tmp_path)
    (root / "caption" / "video-b" / "30sec.json").unlink()
    (root / "visual_memory" / "video-a" / "visual_embeddings.pkl").unlink()

    summary = module.validate_precomputed(eval_json, root, "Qwen3.5-4B")

    assert summary.missing == tuple(sorted(summary.missing))
    assert summary.missing == (
        "caption/video-b/30sec.json",
        "visual_memory/video-a/visual_embeddings.pkl",
    )


def test_validate_precomputed_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    module = load_validator()
    eval_json, root = build_tree(tmp_path)
    eval_json.write_text(
        json.dumps([{"ID": "same", "video_id": "a"}, {"ID": "same", "video_id": "b"}]),
        encoding="utf-8",
    )

    try:
        module.validate_precomputed(eval_json, root, "Qwen3.5-4B")
        raise AssertionError("expected duplicate-ID validation failure")
    except ValueError as exc:
        assert "duplicate question ID" in str(exc)


def test_cli_exits_nonzero_for_incomplete_tree(tmp_path: Path) -> None:
    eval_json, root = build_tree(tmp_path)
    (root / "semantic_memory" / "video-b" / "semantic_consolidation_results_Qwen3.5-4B.json").unlink()

    result = subprocess.run(
        [sys.executable, str(SOURCE), "--eval-json", str(eval_json), "--root", str(root), "--model", "Qwen3.5-4B"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "INCOMPLETE" in result.stderr
    assert "semantic_memory/video-b/semantic_consolidation_results_Qwen3.5-4B.json" in result.stderr
