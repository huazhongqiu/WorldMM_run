import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_run_log():
    source = ROOT / "src" / "worldmm" / "run_log.py"
    spec = importlib.util.spec_from_file_location("worldmm_run_log_presentation", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_question_separator_visibly_delimits_each_question() -> None:
    module = load_run_log()

    assert module.question_separator("2243") == (
        "\n--------------------------------------------------------------\n"
        "QUESTION id=2243"
    )


def test_phrase_weight_averaging_skips_nodes_with_no_selected_facts() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from hipporag.HippoRAG import _average_phrase_weights

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            actual = _average_phrase_weights(
                np.array([8.0, 3.0, 0.0]),
                np.array([2.0, 0.0, 0.0]),
            )

        assert np.array_equal(actual, np.array([4.0, 0.0, 0.0]))
    finally:
        sys.path.remove(str(ROOT / "src"))


def test_benchmark_runners_print_one_answer_event_and_question_separator() -> None:
    for path in (ROOT / "eval" / "eval.py", ROOT / "eval" / "eval_medvidbench.py"):
        source = path.read_text(encoding="utf-8")

        assert "question_separator" in source
        assert 'details["decision"] != "answer"' in source


def test_hipporag_does_not_print_graph_stats_or_progress_bars() -> None:
    hipporag = (ROOT / "src" / "HippoRAG" / "src" / "hipporag" / "HippoRAG.py").read_text(encoding="utf-8")
    embed_utils = (ROOT / "src" / "HippoRAG" / "src" / "hipporag" / "utils" / "embed_utils.py").read_text(encoding="utf-8")

    assert "print(self.get_graph_info())" not in hipporag
    assert "disable=True" in hipporag
    assert "disable=True" in embed_utils
