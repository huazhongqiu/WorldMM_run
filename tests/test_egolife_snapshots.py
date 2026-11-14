import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from worldmm.memory.episodic.memory import EpisodicMemory
from worldmm.memory.episodic.utils import compute_mdhash_id


def memory(tmp_path):
    return EpisodicMemory(SimpleNamespace(text_model_name="embed"), SimpleNamespace(model_name="llm"), None, save_dir_root=str(tmp_path / "worker"))


def test_reuses_matching_30sec_and_retains_cached_coarse_openie(tmp_path):
    m = memory(tmp_path)
    key = compute_mdhash_id("base", prefix="chunk-")
    src = tmp_path / "base.json"
    src.write_text(json.dumps({"ner_results": {key: ["Jake"]}, "triple_results": {key: [["Jake", "holds", "cup"]]}}))
    m.load_precomputed_openie(str(src), base_granularity="30sec", extract_coarse=True)
    target = tmp_path / "openie.json"
    h = SimpleNamespace(openie_results_path=str(target))
    m._seed_precomputed_openie(h, ["base", "missing"], "30sec")
    docs = json.loads(target.read_text())["docs"]
    assert len(docs) == 1
    assert docs[0]["extracted_entities"] == ["Jake"]
    m._seed_precomputed_openie(h, ["base", "missing"], "3min")
    assert json.loads(target.read_text())["docs"] == docs


def test_snapshot_roundtrip_isolated_and_rejects_changed_data(tmp_path, monkeypatch):
    m = memory(tmp_path)
    m.load_captions_from_data({"30sec": [
        {"text": "past", "date": "DAY5", "start_time": "10000000", "end_time": "10003000"},
        {"text": "future", "date": "DAY6", "start_time": "10000000", "end_time": "10003000"}]})
    m.configure_snapshots(str(tmp_path / "snapshots"), load_only=False)
    folder = tmp_path / "worker" / "30sec"
    folder.mkdir(parents=True)
    (folder / "graph.pickle").write_bytes(b"test graph")
    m.hipporag["30sec"] = SimpleNamespace(node_to_node_stats={("a", "b"): 2.0}, ent_node_to_chunk_ids={"a": {"past"}})
    m.indexed_time = 510003000
    m.save_snapshot(510003000)
    saved = tmp_path / "snapshots" / "510003000" / "30sec" / "graph.pickle"
    assert saved.read_bytes() == b"test graph"
    m.configure_snapshots(str(tmp_path / "snapshots"), load_only=True)
    monkeypatch.setattr(m, "_get_or_create_hipporag", lambda g: m.hipporag.setdefault(g, SimpleNamespace()))
    m.index(510003000)
    assert [x.text for x in m.indexed_entries["30sec"]] == ["past"]
    assert m.hipporag["30sec"].ent_node_to_chunk_ids == {"a": {"past"}}
    assert m.hipporag["30sec"].node_to_node_stats == {("a", "b"): 2.0}
    assert m.hipporag["30sec"].read_only_index
    assert Path(m.save_dir_root) == saved.parents[1]
    m._load_snapshot(510003000, for_update=True)
    assert Path(m.save_dir_root) != saved.parents[1]
    Path(m.save_dir_root, "30sec", "graph.pickle").write_bytes(b"worker mutation")
    assert saved.read_bytes() == b"test graph"
    with pytest.raises(FileNotFoundError):
        m.index(610003000)
    m.captions["30sec"][0].text = "changed"
    with pytest.raises(ValueError, match="mismatch"):
        m.index(510003000)


def test_snapshot_corruption_rejected(tmp_path):
    m = memory(tmp_path)
    m.configure_snapshots(str(tmp_path / "snapshots"), load_only=False)
    m.indexed_time = 1
    m.save_snapshot(1)
    metadata = tmp_path / "snapshots" / "1" / "manifest.json"
    doc=json.loads(metadata.read_text()); doc["identity"]["embedding"]="other"; metadata.write_text(json.dumps(doc))
    m.configure_snapshots(str(tmp_path / "snapshots"), load_only=True)
    with pytest.raises(ValueError, match="mismatch"):
        m.index(1)


def test_snapshot_identity_distinguishes_embedding_revisions(tmp_path):
    m = memory(tmp_path)
    m.embedding_model.index_identity = {"backend": "freeret", "weights": "first"}
    before = m._snapshot_identity(1)
    m.embedding_model.index_identity = {"backend": "freeret", "weights": "second"}
    assert before["embedding"] == {"backend": "freeret", "weights": "first"}
    assert before != m._snapshot_identity(1)


def test_preparation_reuses_base_and_only_extracts_missing(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from egolife_precompute import prepare_openie
    m = memory(tmp_path)
    base = tmp_path / "base.json"
    key = compute_mdhash_id("base", prefix="chunk-")
    base.write_text(json.dumps({"ner_results": {key: ["Jake"]}, "triple_results": {key: []}}))
    m.load_precomputed_openie(str(base), base_granularity="30sec", extract_coarse=True)
    m.load_captions_from_data({"30sec": [{"text": "base", "date": "DAY5", "start_time": "10000000", "end_time": "10003000"}],
                               "3min": [{"text": "summary", "date": "DAY5", "start_time": "10000000", "end_time": "10003000"}]})
    calls=[]
    class OpenIE:
        def ner(self, key, text):
            calls.append(text)
            return SimpleNamespace(unique_entities=["Jake"], metadata={})
        def triple_extraction(self, key, text, entities):
            return SimpleNamespace(triples=[], metadata={})
    prepare_openie(m, tmp_path / "prepared", 510003000, extractor=OpenIE())
    prepare_openie(m, tmp_path / "prepared", 510003000, extractor=OpenIE())
    assert calls == ["summary"]
    assert len(json.loads((tmp_path / "prepared" / "30sec.json").read_text())["docs"]) == 1


def test_snapshot_detects_corrupted_file_and_changed_prepared_openie(tmp_path):
    m = memory(tmp_path)
    m.load_captions_from_data({"30sec": [
        {"text": "past", "date": "DAY1", "start_time": "00000000", "end_time": "00000100"},
    ]})
    cutoff = 100000100
    key = compute_mdhash_id("past", prefix="chunk-")
    m.prepared_openie_dir = str(tmp_path / "openie")
    Path(m.prepared_openie_dir).mkdir()
    prepared = Path(m.prepared_openie_dir) / "30sec.json"
    prepared.write_text('{"docs": []}')
    m.configure_snapshots(str(tmp_path / "snapshots"), load_only=False)
    folder = tmp_path / "worker" / "30sec"
    folder.mkdir(parents=True)
    (folder / "graph.pickle").write_bytes(b"graph")
    m.hipporag["30sec"] = SimpleNamespace()
    m.indexed_time = cutoff
    m.save_snapshot(cutoff)
    m.configure_snapshots(str(tmp_path / "snapshots"), load_only=True)
    prepared.write_text(json.dumps({"docs": [{"idx": key, "passage": "past"}]}))
    with pytest.raises(ValueError, match="identity mismatch"):
        m.index(cutoff)
    prepared.write_text('{"docs": []}')
    (tmp_path / "snapshots" / str(cutoff) / "30sec" / "graph.pickle").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="file mismatch"):
        m.index(cutoff)


def test_index_preparation_requires_complete_openie_before_gpu(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from egolife_precompute import prepare_indexes
    m = memory(tmp_path)
    m.prepared_openie_dir = str(tmp_path / "openie")
    Path(m.prepared_openie_dir).mkdir()
    Path(m.prepared_openie_dir, "30sec.json").write_text(json.dumps({"provenance": {"llm": "llm", "base_openie": m._snapshot_identity(1)["base_openie"]}, "docs": []}))
    m.load_captions_from_data({"30sec": [{"text": "missing", "date": "DAY1", "start_time": "00000000", "end_time": "00000100"}]})
    with pytest.raises(ValueError, match="Incomplete OpenIE"):
        prepare_indexes(m, [100000100])


def test_snapshot_identity_uses_visible_openie_content(tmp_path):
    m = memory(tmp_path)
    m.load_captions_from_data({"30sec": [
        {"text": "past", "date": "DAY1", "start_time": "00000000", "end_time": "00000100"},
        {"text": "future", "date": "DAY2", "start_time": "00000000", "end_time": "00000100"},
    ]})
    m.prepared_openie_dir = str(tmp_path / "openie")
    path = Path(m.prepared_openie_dir) / "30sec.json"
    path.parent.mkdir()
    past = {"idx": compute_mdhash_id("past", prefix="chunk-"), "passage": "past",
            "extracted_entities": ["Jake"], "extracted_triples": []}
    path.write_text(json.dumps({"docs": [past]}))
    identity = m._snapshot_identity(100000100)
    future = {"idx": compute_mdhash_id("future", prefix="chunk-"), "passage": "future"}
    path.write_text(json.dumps({"docs": [future, past]}, indent=4))
    assert m._snapshot_identity(100000100) == identity
    past["extracted_entities"] = ["Alice"]
    path.write_text(json.dumps({"docs": [future, past]}, indent=4))
    assert m._snapshot_identity(100000100) != identity
