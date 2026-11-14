import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec = importlib.util.spec_from_file_location('wemm', Path(__file__).parents[1] / 'src/worldmm/embedding/wemm.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_text_shape_normalization_and_lazy_loading(monkeypatch):
    calls = []
    class Model:
        def encode_document(self, content, **kwargs):
            calls.append(content)
            return np.tile([3., 4.], (len(content), 1))
    client = m.WeMMEmbeddingModel(dimensions=2)
    monkeypatch.setattr(client, '_load_model', lambda: Model())
    assert client.encode_text('hello').shape == (1, 2)
    np.testing.assert_allclose(client.encode_text(['a', 'b']), [[.6, .8], [.6, .8]])
    assert calls == [['hello'], ['a', 'b']]
    with pytest.raises(ValueError, match='sparse'):
        client.encode_representations('query')


def test_invalid_dense_rejected(monkeypatch):
    client = m.WeMMEmbeddingModel(dimensions=2)
    class Model:
        def encode_document(self, content, **kwargs):
            return [[float('nan'), 1]]
    monkeypatch.setattr(client, '_load_model', lambda: Model())
    with pytest.raises(ValueError, match='finite'):
        client.encode_text('hello')


def test_query_and_timed_video_document_roles(monkeypatch):
    import sys
    import types
    from PIL import Image
    calls = []
    interval = {'video': '/source.mp4', 'video_start': 2, 'video_end': 4}
    class Decoder:
        def _video_frames(self, spec, **kwargs):
            assert spec == interval
            assert kwargs == {'nframes': 2, 'with_timestamps': True}
            return [Image.new('RGB', (8, 8)), Image.new('RGB', (8, 8))], [2., 3.]
    fake = types.ModuleType('fakeembedding.freeret_api')
    fake.FreeRetAPI = Decoder
    monkeypatch.setitem(sys.modules, 'fakeembedding.freeret_api', fake)
    monkeypatch.setattr(m, '__package__', 'fakeembedding')
    class Model:
        def encode_query(self, documents, **kwargs):
            calls.append(('query', documents))
            return [[1, 0]]
        def encode_document(self, documents, **kwargs):
            calls.append(('document', documents))
            return [[0, 1]]
    client = m.WeMMEmbeddingModel(dimensions=2, nframes=2, max_pixels=16)
    monkeypatch.setattr(client, '_load_model', lambda: Model())
    client.encode_vis_query('find a scene')
    client.encode_video(interval)
    assert calls[0] == ('query', ['find a scene'])
    message = calls[1][1][0][0]
    assert message['role'] == 'user'
    assert [item['text'] for item in message['content'] if item['type'] == 'text'] == ['2.000s', '3.000s']
    assert all(item['image'].size == (4, 4) for item in message['content'] if item['type'] == 'image')
