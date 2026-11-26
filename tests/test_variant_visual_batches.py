import json
import pickle
import sys
import types

import numpy as np

from preprocess import build_variant


class Reader:
    def __init__(self, source, ctx=None, num_threads=1):
        pass

    def get_avg_fps(self):
        return 10.

    def __len__(self):
        return 250

    def get_batch(self, indices):
        return types.SimpleNamespace(asnumpy=lambda: np.zeros((len(indices), 4, 4, 3), np.uint8))


class Encoder:
    backend = 'freeret'
    _freeret = types.SimpleNamespace(concurrency=2)
    index_identity = {'backend': 'fake'}

    def __init__(self):
        self.calls = []

    def encode_representations(self, specs, modality, nframes, batch_size):
        assert modality == 'video' and all(len(spec['video']) == nframes for spec in specs)
        self.calls.append((len(specs), nframes))
        return [{'start': spec['video_start']} for spec in specs]


def test_freeret_visual_clips_are_grouped_by_concurrency_and_frame_count(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'decord', types.SimpleNamespace(VideoReader=Reader, cpu=lambda index: None))
    bounds = [(0, 10), (10, 20), (20, 24), (24, 25)]  # The last clip has only 10 frames.
    base = tmp_path / '10sec.json'
    base.write_text(json.dumps([{'video_path': 'v.mp4', 'start_time': str(start), 'end_time': str(end), 'start_sec': start,
                                 'end_sec': end} for start, end in bounds]))
    encoder = Encoder()
    output = build_variant.visual_index(base, tmp_path / 'visual', encoder, 'videomme_long', 16)
    assert encoder.calls == [(2, 16), (1, 16), (1, 10)]
    with output.open('rb') as stream:
        records = pickle.load(stream)
    samples = json.loads((tmp_path / 'visual' / 'frames.json').read_text())
    assert {key: record['start'] for key, record in records.items()} == {str(s): s for s, _ in bounds}
    assert [samples[str(s)]['nframes'] for s, _ in bounds] == [16, 16, 16, 10]
