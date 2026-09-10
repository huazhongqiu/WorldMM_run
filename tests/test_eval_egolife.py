import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load_eval_module():
    eval_dir = ROOT / "eval"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(eval_dir))
    try:
        spec = importlib.util.spec_from_file_location("worldmm_eval_egolife", eval_dir / "eval_egolife.py")
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(eval_dir))
        sys.path.remove(str(ROOT / "src"))


def test_extract_choice_letter_accepts_final_answer_marker_after_reasoning() -> None:
    module = _load_eval_module()
    response = "I considered the evidence carefully.\nFinal answer: C."
    assert module.extract_choice_letter(response) == "C"


def test_lmdeploy_profile_script_exposes_single_and_dual_gpu_profiles() -> None:
    script = ROOT / "script" / "serve_lmdeploy.sh"
    content = script.read_text(encoding="utf-8")
    assert "debug-1gpu" in content
    assert "formal-2gpu" in content
    assert "--cache-max-entry-count" in content
    assert "--max-batch-size" in content
    assert "--reasoning-parser default" in content
    assert "--tool-call-parser qwen3coder" in content


def test_embedding_device_can_be_separated_from_lmdeploy_gpu(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from worldmm.embedding.embedding_wrapper import EmbeddingModel
        monkeypatch.setenv("WORLDMM_EMBEDDING_DEVICE", "cuda:1")
        assert EmbeddingModel().device == "cuda:1"
    finally:
        sys.path.remove(str(ROOT / "src"))
