import json
import pytest

from preprocess.variant_inputs import align_subtitles, frame_indices, freeze_json, videomme_rows


def test_videomme_keeps_questions_for_video_without_srt(tmp_path):
    (tmp_path / 'with.srt').touch()
    common = {'duration': 'long', 'task_type': 'event', 'question': '?',
              'options': ['A. one', 'B. two', 'C. three', 'D. four'], 'answer': 'B'}
    raw = [{**common, 'videoID': vid, 'question_id': str(i)}
           for i, vid in enumerate(('with', 'without'))]
    rows = videomme_rows(raw, tmp_path)
    assert [(row['ID'], row['video_id'], row['subtitle_path'] is not None) for row in rows] == [
        ('0', 'with', True), ('1', 'without', False),
    ]
    assert rows[1]['choice_b'] == 'two'


def test_subtitles_are_clipped_and_translated_into_source_time():
    cues = [(99., 102., 'first'), (103., 110., 'second'), (105., 108., 'outside')]
    assert align_subtitles(cues, 100., 105., origin=98.) == [
        {'start_time': 2., 'end_time': 4., 'subtitle': 'first'},
        {'start_time': 5., 'end_time': 7., 'subtitle': 'second'},
    ]


def test_frozen_inputs_refuse_changes(tmp_path):
    out = tmp_path / 'questions.json'
    freeze_json(out, [1])
    before = out.stat().st_mtime_ns
    freeze_json(out, [1])
    assert out.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match='Frozen input'):
        freeze_json(out, [2])
    assert json.loads(out.read_text()) == [1]


def test_uniform_samples_respect_exclusive_clip_end():
    assert frame_indices(2, 100, 5, 8, 4) == [10, 12, 13, 15]
    assert frame_indices(2, 100, 5, 5.5, 4) == [10, 10]
