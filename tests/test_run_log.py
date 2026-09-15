import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "worldmm" / "run_log.py"


def load_run_log():
    spec = importlib.util.spec_from_file_location("worldmm_run_log", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_section_uses_major_stage_separator():
    module = load_run_log()

    assert module.section("OFFLINE PREFLIGHT") == (
        "\n"
        + "=" * 64
        + "\n OFFLINE PREFLIGHT\n"
        + "=" * 64
    )


def test_progress_renders_video_level_bar_and_counts():
    module = load_run_log()

    assert module.progress(
        "VIDEO", 3, 20, extra="id=q01CUy_gwdU | questions=18"
    ) == "[VIDEO 003/020] [###" + " " * 17 + "]  15% id=q01CUy_gwdU | questions=18"


def test_agent_round_logs_action_and_simple_parameters():
    module = load_run_log()

    assert module.agent_round(
        2, 5, "search", memory_type="episodic", search_query="who opened the door"
    ) == "round=2/5 | action=search | memory=episodic | query=who opened the door"
    assert module.agent_round(3, 5, "answer") == "round=3/5 | action=answer"


def test_answer_status_logs_state_without_internal_urls():
    module = load_run_log()

    assert module.answer_status(
        "SUCCESS", answer="D", correct=True, elapsed_seconds=2.51,
        tokens=9151, question_id="113",
    ) == "id=113 | status=SUCCESS | answer=D | correct=yes | elapsed=2.5s | tokens=9151"
    assert module.answer_status("ERROR", question_id="114") == "id=114 | status=ERROR"
