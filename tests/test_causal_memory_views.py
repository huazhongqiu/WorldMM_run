import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def visual_module():
    # Load visual code without optional LLM/graph packages.
    name = 'csr_test.visual.memory'
    for package in ('csr_test', 'csr_test.visual'):
        if package not in sys.modules:
            m = ModuleType(package); m.__path__ = [] ; sys.modules[package] = m
    embedding = ModuleType('csr_test.embedding'); embedding.EmbeddingModel = object
    sys.modules['csr_test.embedding'] = embedding
    spec = importlib.util.spec_from_file_location('csr_test.visual.retrieval_scoring', ROOT / 'src/worldmm/memory/visual/retrieval_scoring.py')
    scoring = importlib.util.module_from_spec(spec); sys.modules[spec.name] = scoring; spec.loader.exec_module(scoring)
    # Relative import ...embedding requires a three-level package.
    sys.modules['csr_test.memory'] = sys.modules['csr_test.visual']
    sys.modules['csr_test.memory.visual'] = sys.modules['csr_test.visual']
    sys.modules['csr_test.memory.visual.retrieval_scoring'] = scoring
    spec = importlib.util.spec_from_file_location('csr_test.memory.visual.memory', ROOT / 'src/worldmm/memory/visual/memory.py')
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return m

def test_shuffled_cutoffs_clear_future_and_empty_views():
    m = visual_module()
    memory = m.VisualMemory()
    memory.embedding_lookup = {'a': np.array([1., 0.]), 'b': np.array([0., 1.])}
    memory.load_clips_from_data([dict(id='a', date='DAY1', start_time='10000000', end_time='10003000'), dict(id='b', date='DAY1', start_time='11000000', end_time='11003000')])
    memory.index(111003000)
    assert len(memory.indexed_entries) == 2
    memory.index(110003000)
    assert [c.id for c in memory.indexed_entries] == ['a']
    memory.index(109000000)
    assert memory.indexed_entries == [] and memory.embeddings is None

def test_sparse_refuses_legacy_dense_only(monkeypatch):
    monkeypatch.setenv('WORLDMM_RETRIEVAL_SPARSE', '1')
    m = visual_module()
    memory = m.VisualMemory(SimpleNamespace(encode_vis_query=lambda q: np.array([[1., 0.]])))
    memory.embedding_lookup = {'a': np.array([1., 0.])}
    memory.load_clips_from_data([dict(id='a', date='DAY1', start_time='10000000', end_time='10003000')]); memory.index(110003000)
    with pytest.raises(ValueError, match='sparse|token'):
        memory.retrieve('object', as_context=False)

@pytest.mark.parametrize('center,sparse,rerank', [(bool(i&1),bool(i&2),bool(i&4)) for i in range(8)])
def test_profiles_share_immutable_records_and_calibrate_once(monkeypatch, center, sparse, rerank):
    import hashlib
    import pickle
    for name, value in [('CENTER', center), ('SPARSE', sparse), ('RERANK', rerank)]:
        monkeypatch.setenv('WORLDMM_RETRIEVAL_'+name, str(int(value)))
    monkeypatch.setenv('WORLDMM_DATASET', 'lvbench')
    m = visual_module()
    class Encoder:
        calls = []
        def encode_vis_query(self, value):
            self.calls.append(value)
            if isinstance(value, list):
                return np.tile([[.1, .2]], (len(value), 1))
            return np.array([[1., 0.]])
        def encode_representations(self, value, modality):
            assert modality == 'vis_query'
            self.calls.append(value)
            return [dict(dense=np.array([1.,0.]), token_ids=[7])]
    encoder = Encoder()
    memory = m.VisualMemory(encoder)
    memory.embedding_lookup = {'a': dict(dense=np.array([1.,0.]), token_ids=[1]), 'b': dict(dense=np.array([.1,1.]), token_ids=[7])}
    before = hashlib.sha256(pickle.dumps(memory.embedding_lookup)).hexdigest()
    memory.load_clips_from_data([dict(id='a',date='DAY1',start_time='00000000',end_time='00001000'),dict(id='b',date='DAY1',start_time='00010000',end_time='00011000')])
    memory.index(200000000)
    memory.set_retrieval_context('At 01:05 what is visible?', video_duration=120.)
    first = memory.retrieve('object', top_k=2, as_context=False)
    second = memory.retrieve('object', top_k=2, as_context=False)
    assert len(first) == 2 and [c.id for c in first] == [c.id for c in second]
    assert hashlib.sha256(pickle.dumps(memory.embedding_lookup)).hexdigest() == before
    assert sum(isinstance(call, list) for call in encoder.calls) == int(center)
    if sparse:
        scores = {row['clip_id']: row for row in memory.last_score_trace}
        assert scores['b']['normalized_sparse'] == 1.0
        assert scores['a']['normalized_sparse'] == 0.0
        if not rerank:
            assert scores['a']['score'] == pytest.approx(scores['b']['score'])

def test_timestamp_retrieval_cannot_access_future(monkeypatch):
    m = visual_module(); memory = m.VisualMemory()
    memory.embedding_lookup = {'a': np.array([1., 0.]), 'b': np.array([0., 1.])}
    memory.load_clips_from_data([dict(id='a',date='DAY1',start_time='10000000',end_time='10003000'),dict(id='b',date='DAY1',start_time='11000000',end_time='11003000')])
    memory.index(110003000)
    visited=[]
    monkeypatch.setattr(memory, '_extract_frames', lambda path, **kwargs: visited.append(path) or [])
    memory._retrieve_by_time_range('DAY1 10:59:00 - DAY1 11:05:00')
    assert visited == []

def test_egolife_r_rejects_missing_subtitle_provenance(monkeypatch):
    monkeypatch.setenv('WORLDMM_DATASET','egolife')
    monkeypatch.setenv('WORLDMM_RETRIEVAL_RERANK','1')
    m=visual_module()
    encoder=SimpleNamespace(encode_vis_query=lambda q: np.array([[1.,0.]]))
    memory=m.VisualMemory(encoder)
    memory.embedding_lookup={'a':np.array([1.,0.])}
    entry=dict(id='a',date='DAY1',start_time='10000000',end_time='10003000')
    memory.load_clips_from_data([entry]);memory.index(110003000)
    with pytest.raises(ValueError, match='subtitle'):
        memory.retrieve('object',as_context=False)
    memory.clips=[]
    memory.load_clips_from_data([{**entry,'subtitles':[]}]);memory.reset_index();memory.index(110003000)
    assert len(memory.retrieve('object',as_context=False)) == 1
