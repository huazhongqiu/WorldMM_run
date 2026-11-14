import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from worldmm.embedding import EmbeddingModel


def test_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv('WORLDMM_EMBEDDING_BACKEND', 'typo')
    with pytest.raises(ValueError, match='backend'):
        EmbeddingModel()


def test_original_identity_is_opt_in(monkeypatch):
    monkeypatch.delenv('WORLDMM_EMBEDDING_BACKEND', raising=False)
    assert EmbeddingModel().index_identity is None
    assert EmbeddingModel(backend='original').index_identity['backend'] == 'original'


def test_freeret_routes_every_modality_without_loading_old_models(monkeypatch):
    import worldmm.embedding.freeret_api as api
    calls = []
    class Fake:
        model_name = 'Qwen3.5-4B'
        fingerprint = {'backend': 'freeret', 'model': model_name}
        def encode(self, content, modality='text', **kwargs):
            calls.append(modality)
            return np.ones((1, 2560), dtype=np.float32)
        def encode_representations(self, content, modality='text', **kwargs):
            return [{'dense': self.encode(content, modality)[0], 'token_ids': [1]}]
    monkeypatch.setattr(api, 'FreeRetAPI', lambda **kw: Fake())
    monkeypatch.setenv('WORLDMM_EMBEDDING_BACKEND', 'freeret')
    model = EmbeddingModel(model_config={})
    model.load_model()
    for modality in ['text', 'vis_query', 'image', 'video']:
        assert model.encode('example', modality).shape == (1, 2560)
    assert calls == ['text', 'vis_query', 'image', 'video']
    assert model.text_model is model.vis_model
    assert model.text_model_name == 'Qwen3.5-4B'
    assert model.fingerprint['backend'] == 'freeret'
