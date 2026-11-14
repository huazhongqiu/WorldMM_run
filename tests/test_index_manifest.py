import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('index_manifest', Path(__file__).parents[1] / 'src/worldmm/index_manifest.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_canonical_identity_and_view_separation():
    assert m.fingerprint({'b': 2, 'a': 1}) == m.fingerprint({'a': 1, 'b': 2})
    base = m.index_fingerprint({'model': 'a', 'prompt': 'passage'}, {'frames': [0, 1]})
    assert base != m.index_fingerprint({'model': 'a', 'prompt': 'query'}, {'frames': [0, 1]})
    assert base != m.index_fingerprint({'model': 'a', 'prompt': 'passage'}, {'frames': [1, 0]})
    assert m.view_fingerprint(base, {'center': False}) != m.view_fingerprint(base, {'center': True})


def test_manifest_resume_and_same_text_different_encoder_rejected(tmp_path):
    m.ensure_manifest(tmp_path, {'encoder': 'original', 'prompt': 'v1'})
    m.ensure_manifest(tmp_path, {'encoder': 'original', 'prompt': 'v1'})
    (tmp_path / 'vdb_chunk.parquet').write_bytes(b'same text hash / old vectors')
    with pytest.raises(ValueError, match='identity mismatch'):
        m.ensure_manifest(tmp_path, {'encoder': 'freeret', 'prompt': 'v1'})
    assert (tmp_path / 'vdb_chunk.parquet').read_bytes() == b'same text hash / old vectors'


def test_legacy_nonempty_cache_cannot_claim_new_identity(tmp_path):
    (tmp_path / 'vdb_chunk.parquet').write_bytes(b'old')
    with pytest.raises(ValueError, match='manifest missing'):
        m.ensure_manifest(tmp_path, {'encoder': 'freeret'})


def test_nonfinite_manifest_rejected():
    with pytest.raises(ValueError):
        m.fingerprint({'threshold': float('nan')})


def test_embedding_store_validates_before_loading_same_text_cache(tmp_path):
    """Exercise the real constructor without importing optional retrieval packages."""
    import ast
    import logging
    import os
    source = Path(__file__).parents[1] / 'src/HippoRAG/src/hipporag/embedding_store.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EmbeddingStore')
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__']
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {'os': os, 'logger': logging.getLogger(__name__), 'embedding_identity': m.embedding_identity, 'ensure_manifest': m.ensure_manifest}
    exec(compile(module, str(source), 'exec'), scope)
    Store = scope['EmbeddingStore']
    loaded = []
    Store._load_data = lambda self: loaded.append(self.filename)
    class Encoder:
        index_identity = {'model': 'original', 'prompt': 'v1'}
    Store(Encoder(), str(tmp_path), 1, 'chunk')
    (tmp_path / 'vdb_chunk.parquet').write_bytes(b'old-vector-same-text-id')
    Encoder.index_identity = {'model': 'freeret', 'prompt': 'v1'}
    with pytest.raises(ValueError, match='identity mismatch'):
        Store(Encoder(), str(tmp_path), 1, 'chunk')
    assert len(loaded) == 1
