"""Content identities shared by embedding indexes and derived retrieval views."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(stream.name, path)


def index_fingerprint(embedding, inputs):
    """Retrieval switches intentionally do not belong in the base identity."""
    return fingerprint({'embedding': embedding, 'inputs': inputs})


def view_fingerprint(base_fingerprint, retrieval):
    return fingerprint({'index': base_fingerprint, 'retrieval': retrieval})


def embedding_identity(model):
    """Legacy encoders opt out; explicitly configured backends must supply identity."""
    has_identity = hasattr(model, 'index_identity')
    identity = getattr(model, 'index_identity', None)
    if callable(identity):
        identity = identity()
    if identity is not None:
        if not isinstance(identity, dict) or not identity:
            raise ValueError('Embedding index_identity must be a nonempty dictionary')
        return identity
    if has_identity:
        return None
    value = getattr(model, 'fingerprint', None)
    if callable(value):
        value = value()
    return {'embedding_fingerprint': value} if value else None


def ensure_manifest(directory, identity):
    """Refuse unidentified or incompatible artifacts before any vector/graph load."""
    directory = Path(directory)
    path = directory / 'index_manifest.json'
    expected = {'schema_version': 1, 'identity': identity, 'fingerprint': fingerprint(identity)}
    if path.exists():
        actual = json.loads(path.read_text(encoding='utf-8'))
        if actual != expected:
            raise ValueError(f'Index identity mismatch: {path}')
        return expected
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Index manifest missing in nonempty cache: {directory}; use a new namespace')
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing another writer's identity.
    try:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(canonical_json(expected) + '\n')
    except FileExistsError:
        return ensure_manifest(directory, identity)
    return expected
