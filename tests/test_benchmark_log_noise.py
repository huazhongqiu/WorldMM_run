import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    sys.path.insert(0, str(ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "src"))


def test_lvbench_console_logging_keeps_runner_info_and_silences_dependencies() -> None:
    module = load_module("worldmm_lvbench_log_noise", ROOT / "eval" / "eval.py")

    module.configure_console_logging()

    assert module.logger.level == logging.INFO
    for logger_name in ("hipporag", "sentence_transformers", "transformers", "worldmm.memory", "worldmm.embedding"):
        assert logging.getLogger(logger_name).level == logging.ERROR
    assert module.os.environ["TQDM_DISABLE"] == "1"


def test_medvidbench_console_logging_keeps_runner_info_and_silences_dependencies() -> None:
    module = load_module("worldmm_medvidbench_log_noise", ROOT / "eval" / "eval_medvidbench.py")

    module.configure_logging()

    assert module.logger.level == logging.INFO
    for logger_name in ("hipporag", "sentence_transformers", "transformers", "worldmm.memory", "worldmm.embedding"):
        assert logging.getLogger(logger_name).level == logging.ERROR
    assert module.os.environ["TQDM_DISABLE"] == "1"


def test_qwen3_embedding_disables_sentence_transformer_progress_bar() -> None:
    source = (ROOT / "src" / "worldmm" / "embedding" / "qwen3_embedding.py").read_text(encoding="utf-8")

    assert "show_progress_bar=False" in source
