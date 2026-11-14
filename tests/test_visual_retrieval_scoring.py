import importlib.util
from pathlib import Path
import numpy as np
import pytest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

PATH = Path(__file__).resolve().parents[1] / 'src/worldmm/memory/visual/retrieval_scoring.py'

def scorer():
    assert PATH.exists(), 'C/S/R scoring module must exist'
    spec = importlib.util.spec_from_file_location('visual_scoring', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_sparse_scores_entire_pool_and_deduplicates():
    m = scorer()
    scores = m.bm25_sets([3, 3], [[1], [3, 3], [3, 4]])
    assert scores[1] > scores[2] > scores[0]
    assert np.allclose(scores, m.bm25_sets([3], [[1], [3], [3, 4]]))

def test_center_uses_visible_pool_and_degenerate_is_finite():
    m = scorer()
    vectors = np.array([[1., 0.], [0., 1.]])
    raw, centered = m.dense_scores([1., 0.], vectors, center=True, query_mean=[0., 0.], visible_vectors=vectors)
    assert centered[0] > centered[1]
    assert np.isfinite(m.dense_scores([0., 0.], vectors[:1], center=True, query_mean=[0., 0.], visible_vectors=vectors[:1])[1]).all()

def test_videomme_rejects_r_and_all_flags_parse_explicitly(monkeypatch):
    m = scorer()
    monkeypatch.setenv('WORLDMM_RETRIEVAL_RERANK', '1')
    monkeypatch.setenv('WORLDMM_DATASET', 'videomme_long')
    with pytest.raises(ValueError, match='Video-MME'):
        m.retrieval_config()
    monkeypatch.setenv('WORLDMM_RETRIEVAL_RERANK', 'maybe')
    with pytest.raises(ValueError):
        m.retrieval_config()

def test_lvbench_named_time_scores_every_candidate_and_ignores_options():
    m = scorer()
    result = m.rerank_scores([1., 0.], [{'start': 0., 'end': 10.}, {'start': 100., 'end': 110.}], dataset='lvbench', question='At 01:45 what happens?\n(A) 00:01', query_time=None, duration=120.)
    assert result[1] > .99
    assert result[0] < 1.001

def test_egolife_wall_clock_nms():
    m = scorer()
    order = m.select_indices([1., .9, .8], 3, dataset='egolife', rerank=True, ends=[10., 50., 100.])
    assert order == [0, 2]

def test_videospy_config_overlay_drives_flags_and_rejects_unsupported(tmp_path, monkeypatch):
    m = scorer()
    base = tmp_path/'default.yml'; base.write_text('tools:\n  semantic_search:\n    index_method: single\n    single:\n      center_embeddings: false\n      keyword_search: false\n')
    overlay = tmp_path/'lvbench_retrieval_best.yml'; overlay.write_text('tools:\n  semantic_search:\n    single:\n      center_embeddings: true\n      keyword_search: true\n    lvbench_rerank:\n      time_weight: 1.0\n      time_tau: 10\n      opening_weight: 0.5\n')
    monkeypatch.setenv('WORLDMM_RETRIEVAL_CONFIG', str(overlay))
    monkeypatch.setenv('WORLDMM_DATASET', 'lvbench')
    flags, dataset = m.retrieval_config()
    assert flags['center'] and flags['sparse'] and flags['rerank']
    overlay.write_text('tools:\n  semantic_search:\n    index_method: multigranular\n')
    with pytest.raises(ValueError, match='single'):
        m.retrieval_config()
