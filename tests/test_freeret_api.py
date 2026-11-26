import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from worldmm.embedding.freeret_api import FreeRetAPI


def response():
    return {'model': 'Qwen3.5-4B', 'embedding_method': 'freeret_pre_ffn_v1',
            'distribution_version': 'next_token_full_softmax_v1', 'tokenizer_fingerprint': 'tok',
            'data': [{'index': 0, 'embedding': [1.] * 2560, 'sparse_tokens': [
                {'token_id': i, 'token': str(i), 'probability': .001} for i in range(512)]}]}


def client(handler=None, concurrency=None):
    return FreeRetAPI(concurrency=concurrency, client=httpx.Client(transport=httpx.MockTransport(
        handler or (lambda request: httpx.Response(200, json=response())))))


def test_shapes_order_and_raw_probabilities():
    import json
    seen = []
    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        result = response()
        result['data'][0]['embedding'][0] = 2. if 'Text: b' in body['messages'][-1]['content'][0]['text'] else 1.
        return httpx.Response(200, json=result)
    api = client(handle)
    reps = api.encode_representations(['a', 'b'])
    assert len(reps) == 2
    assert reps[1]['dense'][0] > reps[0]['dense'][0]
    assert reps[0]['token_probs'] == [.001] * 512
    assert api.encode('a').shape == (1, 2560)
    assert api.encode([]).shape == (0, 2560)
    assert all(x['token_top_k'] == 512 for x in seen)
    assert 'a' in seen[0]['messages'][-1]['content'][0]['text']


@pytest.mark.parametrize('damage', ['dim', 'nan', 'zero', 'missing', 'version', 'model', 'prob', 'duplicate'])
def test_rejects_invalid_response(damage):
    result = response()
    item = result['data'][0]
    if damage == 'dim': item['embedding'] = [1.]
    if damage == 'nan': item['embedding'][0] = 'NaN'
    if damage == 'zero': item['embedding'] = [0.] * 2560
    if damage == 'missing': item.pop('sparse_tokens')
    if damage == 'version': result['distribution_version'] = 'other'
    if damage == 'model': result['model'] = 'other'
    if damage == 'prob': item['sparse_tokens'][0]['probability'] = 2
    if damage == 'duplicate': item['sparse_tokens'][0]['token_id'] = 1
    with pytest.raises(ValueError):
        client(lambda req: httpx.Response(200, json=result)).encode('a')


def test_images_video_and_explicit_roles(monkeypatch):
    import json
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    api = client(handle)
    monkeypatch.setattr(api, '_video_frames', lambda spec, **kw: ([Image.new('RGB', (2, 2))] * 2, [1., 3.]))
    api.encode_image(Image.new('RGB', (2, 2)))
    api.encode_video({'video': '/a.mp4', 'video_start': 1, 'video_end': 3})
    api.encode_text('a', role='query')
    api.encode_vis_query(['a'])
    assert sum(c['type'] == 'image_url' for c in seen[1]['messages'][-1]['content']) == 2
    assert seen[2]['messages'] == seen[3]['messages']


def test_failure_does_not_poison_subsequent_request():
    count = 0
    def handle(request):
        nonlocal count
        count += 1
        return httpx.Response(503) if count == 1 else httpx.Response(200, json=response())
    api = client(handle)
    with pytest.raises(httpx.HTTPStatusError): api.encode('a')
    assert api.encode('a').shape == (1, 2560)


def test_stale_keepalive_disconnect_is_resent_but_bounded():
    count = 0
    def handle(request):
        nonlocal count
        count += 1
        if count <= 2:
            raise httpx.RemoteProtocolError('Server disconnected without sending a response.')
        return httpx.Response(200, json=response())
    assert client(handle).encode('a').shape == (1, 2560) and count == 3
    def always(request):
        raise httpx.RemoteProtocolError('Server disconnected without sending a response.')
    with pytest.raises(httpx.RemoteProtocolError): client(always).encode('a')


def test_source_time_video_frames_preserve_order(tmp_path):
    cv2 = pytest.importorskip('cv2')
    path = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (16, 16))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((16, 16, 3), i * 10, dtype=np.uint8))
    writer.release()
    frames = client()._video_frames({'video': str(path), 'video_start': .5, 'video_end': 1.5}, nframes=3)
    np.testing.assert_allclose([np.asarray(f).mean() for f in frames], [50, 100, 140], atol=3)


def test_batch_failure_never_returns_partial_vectors():
    count = 0
    def handle(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json=response()) if count == 1 else httpx.Response(503)
    with pytest.raises(httpx.HTTPStatusError):
        client(handle).encode(['first', 'second'])


def test_scalar_and_list_have_same_passage_prompt():
    import json
    requests = []
    def handle(request):
        requests.append(json.loads(request.content)['messages'])
        return httpx.Response(200, json=response())
    api = client(handle)
    api.encode_text('same')
    api.encode_text(['same'])
    assert requests[0] == requests[1]


def test_deployment_identity_and_config_overrides():
    with pytest.raises(ValueError, match='verified'):
        client().index_identity
    api = FreeRetAPI(model_config={'model_name': 'openai/Qwen3.5-4B', 'api_base': 'http://service/v1',
                                  'max_image_pixels': 262144, 'embedding_method': 'freeret'},
                     model_fingerprint='weights', tokenizer_fingerprint='tok', processor_fingerprint='proc')
    assert api.index_identity['model'] == 'Qwen3.5-4B'
    assert api.index_identity['max_pixels'] == 262144
    assert api.index_identity['prompts']['vis_query'].startswith('Task: Represent a text query')


def test_custom_query_template_is_in_identity_and_request():
    import json
    seen = []
    api = FreeRetAPI(prompt_templates={'vis_query': 'Custom {query}'}, client=httpx.Client(transport=
        httpx.MockTransport(lambda request: seen.append(json.loads(request.content)) or httpx.Response(200, json=response()))))
    api.encode_vis_query('question')
    assert seen[0]['messages'][-1]['content'][0]['text'] == 'Custom question'
    assert api.fingerprint['prompts']['vis_query'] == 'Custom {query}'


def test_concurrent_requests_overlap_and_keep_input_order():
    import json
    import threading
    barrier = threading.Barrier(2, timeout=5)
    def handle(request):
        text = json.loads(request.content)['messages'][-1]['content'][0]['text']
        barrier.wait()  # Fails unless two requests are in flight together.
        result = response()
        result['data'][0]['embedding'][0] = float(text.split('Text: item')[1][0]) + 1
        return httpx.Response(200, json=result)
    reps = client(handle, concurrency=2).encode_representations([f'item{i}' for i in range(4)])
    assert [rep['dense'][0] for rep in reps] == sorted(rep['dense'][0] for rep in reps)


def test_concurrent_failure_fails_whole_call():
    import json
    def handle(request):
        failed = 'Text: bad' in json.loads(request.content)['messages'][-1]['content'][0]['text']
        return httpx.Response(503) if failed else httpx.Response(200, json=response())
    with pytest.raises(httpx.HTTPStatusError):
        client(handle, concurrency=4).encode(['a', 'bad', 'c', 'd'])


def test_concurrency_is_transport_not_identity(monkeypatch):
    fingerprints = dict(model_fingerprint='weights', tokenizer_fingerprint='tok', processor_fingerprint='proc')
    assert FreeRetAPI(concurrency=1, **fingerprints).index_identity == FreeRetAPI(concurrency=8, **fingerprints).index_identity
    monkeypatch.setenv('WORLDMM_FREERET_CONCURRENCY', '3')
    assert FreeRetAPI().concurrency == 3
    with pytest.raises(ValueError):
        FreeRetAPI(concurrency=0)
