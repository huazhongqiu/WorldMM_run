from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_qwen3_embedding_default_batch_size_is_memory_safe() -> None:
    source = (ROOT / "src" / "worldmm" / "embedding" / "qwen3_embedding.py").read_text(encoding="utf-8")

    assert "batch_size: int = 64" in source


def test_episodic_hipporag_uses_memory_safe_embedding_batch(monkeypatch) -> None:
    from worldmm.memory.episodic import memory as episodic_module
    from worldmm.memory.episodic.memory import EpisodicMemory

    class FakeHippoRAG:
        def __init__(self, **_kwargs):
            self.global_config = SimpleNamespace(embedding_batch_size=128)

    monkeypatch.setattr(episodic_module, "HippoRAG", FakeHippoRAG)
    memory = EpisodicMemory(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), granularities=["30sec"])

    assert memory._get_or_create_hipporag("30sec").global_config.embedding_batch_size == 64


def test_episodic_index_releases_cuda_cache_after_each_granularity(monkeypatch) -> None:
    from worldmm.memory.episodic import memory as episodic_module
    from worldmm.memory.episodic.memory import EpisodicMemory

    class FakeHippoRAG:
        def __init__(self):
            self.openie = SimpleNamespace()

        def update(self, docs):
            assert docs

    cleared = []
    monkeypatch.setattr(episodic_module, "_clear_cuda_cache", lambda: cleared.append(True))
    memory = EpisodicMemory(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), granularities=["10sec", "30sec"])
    for granularity in memory.granularities:
        memory.captions[granularity] = [SimpleNamespace(timestamp_int=(0, 1), text=f"{granularity} caption")]
    memory._get_or_create_hipporag = lambda _granularity: FakeHippoRAG()

    memory.index(1)

    assert cleared == [True, True]


def test_benchmark_runners_emit_per_video_memory_index_lifecycle_logs() -> None:
    for path in (ROOT / "eval" / "eval.py", ROOT / "eval" / "eval_medvidbench.py"):
        source = path.read_text(encoding="utf-8")
        assert "building memory index..." in source
        assert "index done in" in source
        assert "index=FAILED" in source
