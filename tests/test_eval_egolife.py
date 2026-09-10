import json
import threading
import time

import pytest
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


def test_smoke_script_defaults_to_flash_attention_2() -> None:
    script = ROOT / "script" / "run_egolife_smoke.sh"
    content = script.read_text(encoding="utf-8")
    assert 'WORLDMM_TEXT_ATTENTION="${WORLDMM_TEXT_ATTENTION:-flash_attention_2}"' in content
    assert 'WORLDMM_VLM_ATTENTION="${WORLDMM_VLM_ATTENTION:-flash_attention_2}"' in content


def test_question_manifest_selects_exact_ids_in_manifest_order(tmp_path: Path) -> None:
    module = _load_eval_module()
    manifest = tmp_path / "test.json"
    manifest.write_text(json.dumps(["3", "1"]), encoding="utf-8")
    rows = [{"ID": "1"}, {"ID": "2"}, {"ID": "3"}]

    assert module.select_questions(rows, manifest) == [{"ID": "3"}, {"ID": "1"}]

    manifest.write_text(json.dumps(["4"]), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown EgoLife question IDs"):
        module.select_questions(rows, manifest)


def test_parallel_evaluator_preserves_input_order_and_uses_multiple_threads() -> None:
    module = _load_eval_module()
    thread_ids: set[int] = set()
    lock = threading.Lock()

    def evaluate(item: int) -> int:
        with lock:
            thread_ids.add(threading.get_ident())
        time.sleep(0.02)
        return item * 10

    assert module.evaluate_in_parallel([1, 2, 3, 4], workers=2, evaluate=evaluate) == [10, 20, 30, 40]
    assert len(thread_ids) >= 2


def test_test_runner_uses_the_committed_day6_day7_manifest() -> None:
    script = ROOT / "script" / "run_egolife_test.sh"
    content = script.read_text(encoding="utf-8")
    assert 'eval/splits/egolife_a1_jake_day6_day7_test.json' in content
    assert '--workers "${WORKERS}"' in content


def test_select_evaluation_rows_applies_manifest_before_limit(tmp_path: Path) -> None:
    module = _load_eval_module()
    manifest = tmp_path / "test.json"
    manifest.write_text(json.dumps(["3", "1"]), encoding="utf-8")
    rows = [{"ID": "1"}, {"ID": "2"}, {"ID": "3"}]

    assert module.select_evaluation_rows(rows, manifest, limit=1) == [{"ID": "3"}]


def test_synchronized_embedding_model_locks_generic_encode() -> None:
    module = _load_eval_module()

    class FakeEmbeddingModel:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def encode(self, item: int) -> int:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            return item

    fake = FakeEmbeddingModel()
    synchronized = module.SynchronizedEmbeddingModel(fake)

    assert module.evaluate_in_parallel([1, 2], workers=2, evaluate=synchronized.encode) == [1, 2]
    assert fake.max_active == 1


def test_test_runner_defaults_to_four_workers() -> None:
    script = ROOT / "script" / "run_egolife_test.sh"
    content = script.read_text(encoding="utf-8")
    assert 'WORLDMM_EGOLIFE_WORKERS:-4' in content
