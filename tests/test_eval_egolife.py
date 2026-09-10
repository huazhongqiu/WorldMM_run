import json
import logging
import threading
import time
from types import SimpleNamespace

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


def test_workflow_logger_emits_readable_worker_round_and_answer_events() -> None:
    module = _load_eval_module()
    lines: list[str] = []
    progress = module.WorkflowProgressLogger(lines.append)

    progress.start_flow("Worker 2 题目开始", worker=2, question_id="406", run="1/95")
    progress.round(
        worker=2,
        question_id="406",
        round_num=1,
        decision="search",
        memory_type="episodic",
        search_query="where did the person put the keys",
        top_k=3,
    )
    progress.answer(
        worker=2,
        question_id="406",
        answer="B",
        rounds=2,
        total_tokens=1234,
        elapsed_seconds=4.5,
    )

    rendered = "\n".join(lines)
    assert "========== Worker 2 题目开始 ==========" in rendered
    assert "worker=2 question_id=406 run=1/95" in rendered
    assert "round=1 action=search memory=episodic top_k=3" in rendered
    assert "answer=B rounds=2 tokens=1234 elapsed=4.50s" in rendered


def test_console_logging_suppresses_http_request_noise() -> None:
    module = _load_eval_module()
    module.configure_console_logging()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert module.logger.level == logging.WARNING
    assert logging.getLogger("worldmm.memory").level == logging.ERROR


def test_test_runner_defaults_to_worldmm_egolife_output_root() -> None:
    script = ROOT / "script" / "run_egolife_test.sh"
    content = script.read_text(encoding="utf-8")
    assert 'WORLDMM_EGOLIFE_OUTPUT_ROOT:-/myworkspace/projects/output/worldmm/egolife' in content



def test_test_runner_disables_dependency_progress_bars() -> None:
    script = ROOT / "script" / "run_egolife_test.sh"
    assert "export TQDM_DISABLE=1" in script.read_text(encoding="utf-8")



def test_parallel_evaluator_passes_worker_context_to_memory_progress_callback() -> None:
    content = (ROOT / "eval" / "eval_egolife.py").read_text(encoding="utf-8")
    assert "progress_callback=on_memory_progress" in content
    assert "worker=state.worker_id" in content
    assert "workflow.start_flow(" in content
    assert "progress.close()" not in content


def test_progress_bar_renders_completed_fraction() -> None:
    module = _load_eval_module()

    assert module.render_progress_bar(0, 10, width=10) == "[----------] 0.0% (0/10)"
    assert module.render_progress_bar(5, 10, width=10) == "[#####-----] 50.0% (5/10)"
    assert module.render_progress_bar(12, 10, width=10) == "[##########] 100.0% (10/10)"


def test_episodic_index_wires_openie_progress_callback() -> None:
    content = (ROOT / "src" / "worldmm" / "memory" / "episodic" / "memory.py").read_text(encoding="utf-8")
    assert "progress_callback" in content
    assert "set_progress_callback" in content


def test_openie_reports_throttled_ner_and_triple_progress(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from worldmm.memory.episodic.openie import OpenIE

        openie = OpenIE(SimpleNamespace(model_name="test-model"))
        events: list[tuple[str, int, int]] = []
        openie.set_progress_callback(lambda stage, completed, total: events.append((stage, completed, total)))
        openie.ner = lambda chunk_id, _: SimpleNamespace(chunk_id=chunk_id, unique_entities=[])
        openie.triple_extraction = lambda chunk_id, _passage, _entities: SimpleNamespace(chunk_id=chunk_id, triples=[])

        openie.batch_openie(["first", "second"], output_dir=str(tmp_path))

        assert ("ner", 0, 2) in events
        assert ("ner", 2, 2) in events
        assert ("triples", 0, 2) in events
        assert ("triples", 2, 2) in events
    finally:
        sys.path.remove(str(ROOT / "src"))
