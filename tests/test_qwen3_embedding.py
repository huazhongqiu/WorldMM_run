import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


class FakeSentenceTransformer:
    kwargs = None

    def __init__(self, _model_name, **kwargs):
        type(self).kwargs = kwargs


sys.modules["sentence_transformers"] = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
SOURCE = Path(__file__).resolve().parents[1] / "src" / "worldmm" / "embedding" / "qwen3_embedding.py"
SPEC = importlib.util.spec_from_file_location("worldmm_qwen3_embedding", SOURCE)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_qwen3_embedding_uses_transformers_452_compatible_load_kwargs() -> None:
    MODULE.Qwen3EmbeddingModel(model_name="/models/Qwen3-Embedding-4B", device="cuda")
    model_kwargs = FakeSentenceTransformer.kwargs["model_kwargs"]
    assert model_kwargs["torch_dtype"] == "auto"
    assert "dtype" not in model_kwargs
    assert FakeSentenceTransformer.kwargs["processor_kwargs"] == {"padding_side": "left"}
