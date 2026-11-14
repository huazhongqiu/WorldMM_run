"""Qwen3.5 FreeRet transport; sorting and C/S/R belong to memory retrieval."""
import base64
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from .freeret_prompts import (IMAGE_TOKEN, QUERY_SEARCH_TEMPLATE, CLIP_SEARCH_TEMPLATE,
                              TEXT_SEARCH_TEMPLATE, _clip_evidence)


class FreeRetAPI:
    dimension = 2560
    feature_version = 'freeret_pre_ffn_v1'
    distribution_version = 'next_token_full_softmax_v1'
    prompts = {'passage': TEXT_SEARCH_TEMPLATE, 'query': QUERY_SEARCH_TEMPLATE,
               'vis_query': QUERY_SEARCH_TEMPLATE, 'video': CLIP_SEARCH_TEMPLATE}

    def __init__(self, base_url=None, model_name=None, api_key=None, client=None,
                 timeout=300, nframes=16, max_pixels=360 * 420,
                 model_fingerprint=None, processor_fingerprint=None, tokenizer_fingerprint=None,
                 model_config=None, prompt_templates=None):
        config = dict(model_config or {})
        base_url = config.get('api_base', base_url)
        model_name = config.get('model_name', model_name)
        if model_name:
            model_name = model_name.removeprefix('openai/')
        max_pixels = config.get('max_image_pixels', max_pixels)
        api_key = config.get('api_key', api_key)
        timeout = config.get('timeout', timeout)
        if config.get('embedding_method', 'freeret') != 'freeret':
            raise ValueError('FreeRet backend requires embedding_method=freeret in retrieval config')
        self.prompts = dict(type(self).prompts)
        for key, template in (prompt_templates or {}).items():
            if key not in self.prompts or not isinstance(template, str):
                raise ValueError('Invalid FreeRet prompt template override')
            self.prompts[key] = template
        self.base_url = (base_url or os.environ.get('WORLDMM_FREERET_API_BASE', 'http://127.0.0.1:23333/v1')).rstrip('/')
        self.model_name = model_name or os.environ.get('WORLDMM_FREERET_MODEL', 'Qwen3.5-4B')
        self.api_key = api_key or os.environ.get('WORLDMM_FREERET_API_KEY', 'EMPTY')
        self.client = client if client is not None else httpx.Client(timeout=timeout)
        if nframes <= 0 or max_pixels <= 0:
            raise ValueError('nframes and max_pixels must be positive')
        self.nframes, self.max_pixels = nframes, max_pixels
        # These are explicit deployment identities, never credentials or transient C/S/R settings.
        self.fingerprint = {
            'backend': 'freeret', 'model': self.model_name, 'endpoint': self.base_url,
            'model_fingerprint': model_fingerprint or os.environ.get('WORLDMM_FREERET_MODEL_FINGERPRINT', 'UNVERIFIED'),
            'processor_fingerprint': processor_fingerprint or os.environ.get('WORLDMM_FREERET_PROCESSOR_FINGERPRINT', 'UNVERIFIED'),
            'tokenizer_fingerprint': tokenizer_fingerprint or os.environ.get('WORLDMM_FREERET_TOKENIZER_FINGERPRINT', 'UNVERIFIED'),
            'patch_commit': '9a4aecd70115770e80133a73a9cca1973060722e',
            'feature_version': self.feature_version, 'distribution_version': self.distribution_version,
            'dimension': self.dimension, 'dtype': 'float32', 'normalization': 'l2',
            'prompts': dict(self.prompts), 'prompt_version': 'videospy_one_word_20261114_v1',
            'frame_sampling': 'uniform_source_frames_inclusive_v1', 'nframes': nframes,
            'max_pixels': max_pixels, 'image_encoding': 'RGB_JPEG_quality95_v1', 'token_top_k': 512,
        }
        self._observed_tokenizer = None

    @property
    def index_identity(self):
        if any(self.fingerprint[key] == 'UNVERIFIED' for key in
               ('model_fingerprint', 'processor_fingerprint', 'tokenizer_fingerprint')):
            raise ValueError('FreeRet persistence requires verified deployment fingerprints')
        return self.fingerprint

    def _image_part(self, image, max_pixels):
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return self._image_part(opened, max_pixels)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        rgb = image.convert('RGB')
        if rgb.width * rgb.height > max_pixels:
            scale = (max_pixels / (rgb.width * rgb.height)) ** .5
            rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))))
        with BytesIO() as buffer:
            rgb.save(buffer, format='JPEG', quality=95)
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        return {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + encoded}}

    def _video_frames(self, spec, *, nframes, with_timestamps=False, **kwargs):
        """Decode a WorldMM source-time interval, never send a filename as text."""
        import cv2
        spec = {'video': str(spec)} if isinstance(spec, (str, Path)) else dict(spec)
        if 'frame_timestamps' in spec:
            frames, timestamps = spec['video'], spec['frame_timestamps']
            if len(frames) != nframes or len(timestamps) != nframes:
                raise ValueError('Predecoded frame count does not match the requested budget')
            return (frames, timestamps) if with_timestamps else frames
        if 'video' not in spec:
            raise ValueError("Video dictionaries must contain 'video'")
        capture = cv2.VideoCapture(str(spec['video']))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not capture.isOpened() or not np.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError(f"Cannot decode video: {spec['video']}")
            start = float(spec.get('video_start', 0))
            end = float(spec.get('video_end', count / fps))
            if not np.isfinite([start, end]).all() or start < 0 or end <= start or end > count / fps + 1 / fps:
                raise ValueError('Invalid source video interval')
            first = int(np.ceil(start * fps))
            last = min(count - 1, int(np.ceil(end * fps)) - 1)
            if last < first:
                raise ValueError('Video interval contains no frames')
            indices = np.linspace(first, last, nframes).round().astype(int)
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Failed decoding frame {index}')
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return (frames, [float(i / fps) for i in indices]) if with_timestamps else frames
        finally:
            capture.release()

    def _request(self, content, *, max_pixels):
        result = self.client.post(self.base_url + '/embeddings', headers={'Authorization': 'Bearer ' + self.api_key}, json={
            'model': self.model_name, 'embedding_method': 'freeret', 'encoding_format': 'float',
            'token_top_k': 512, 'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': content}],
            'mm_processor_kwargs': {'image': {'min_pixels': min(65536, max_pixels), 'max_pixels': max_pixels}},
        })
        result.raise_for_status()
        value = result.json()
        if (value.get('model') != self.model_name or value.get('embedding_method') != self.feature_version
                or value.get('distribution_version') != self.distribution_version):
            raise ValueError('Incompatible FreeRet model/feature/distribution response')
        data = value.get('data', [])
        if len(data) != 1 or data[0].get('index', 0) != 0:
            raise ValueError('FreeRet must return exactly one ordered representation')
        dense = np.asarray(data[0].get('embedding'), dtype=np.float32)
        norm = np.linalg.norm(dense)
        if dense.shape != (self.dimension,) or not np.isfinite(dense).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError('FreeRet requires a finite nonzero 2560-dimensional embedding')
        terms = data[0].get('sparse_tokens', [])
        if len(terms) != 512:
            raise ValueError('FreeRet requires top-512 sparse_tokens')
        ids = [term.get('token_id') for term in terms]
        probs = [term.get('probability') for term in terms]
        tokens = [term.get('token') for term in terms]
        if (any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids)
                or any(not isinstance(t, str) for t in tokens)):
            raise ValueError('Invalid sparse token IDs/text')
        try:
            probability = np.asarray(probs, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid sparse probabilities') from exc
        if (not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()
                or probability.sum() > 1 + 1e-5 or (np.diff(probability) > 1e-8).any()):
            raise ValueError('Expected ranked raw full-softmax sparse probabilities')
        tokenizer = value.get('tokenizer_fingerprint')
        expected = self.fingerprint['tokenizer_fingerprint']
        if not isinstance(tokenizer, str) or not tokenizer or (expected != 'UNVERIFIED' and tokenizer != expected):
            raise ValueError('Missing or mismatched tokenizer fingerprint')
        if self._observed_tokenizer is not None and tokenizer != self._observed_tokenizer:
            raise ValueError('Tokenizer changed during encoding')
        self._observed_tokenizer = tokenizer
        return {'dense': dense / norm, 'token_ids': ids, 'token_probs': probs, 'tokens': tokens,
                'metadata': {'feature_version': self.feature_version, 'distribution_version': self.distribution_version,
                             'tokenizer_fingerprint': tokenizer, 'retained_probability_mass': float(probability.sum()),
                             'identity': self.fingerprint}}

    def encode_representations(self, content, modality='text', *, role='passage', batch_size=16,
                               query_text=None, nframes=None, max_pixels=None, subtitles=None):
        if modality not in ('text', 'vis_query', 'image', 'video'):
            raise ValueError(f'Unsupported modality: {modality}')
        if role not in ('passage', 'query'):
            raise ValueError('Text role must be passage or query')
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        frames_count = self.nframes if nframes is None else nframes
        pixels = self.max_pixels if max_pixels is None else max_pixels
        if type(frames_count) is not int or frames_count <= 0 or type(pixels) is not int or pixels <= 0:
            raise ValueError('nframes/max_pixels must be positive integers')
        values = content if isinstance(content, list) else [content]
        results = []
        # Endpoint messages represent one item. Sequential requests preserve order and
        # fail the entire call rather than returning partial or fabricated embeddings.
        for value in values:
            if modality in ('text', 'vis_query'):
                if not isinstance(value, str):
                    raise TypeError('Text embeddings require strings')
                template = self.prompts['vis_query' if modality == 'vis_query' else role]
                prompt = template.format(text=value, query=value)
                parts = [{'type': 'text', 'text': prompt}]
            else:
                if modality == 'image':
                    frames, timestamps, start = [value], [0.], 0.
                else:
                    frames, timestamps = self._video_frames(value, nframes=frames_count, with_timestamps=True)
                    start = float(value.get('video_start', 0)) if isinstance(value, dict) else 0.
                evidence, subtitle_section = _clip_evidence(timestamps, subtitles or [], start)
                prompt = query_text or self.prompts['video'].format(frames=evidence, subtitle_section=subtitle_section)
                pieces = prompt.split(IMAGE_TOKEN)
                if len(pieces) != len(frames) + 1:
                    raise ValueError('Visual prompt image placeholders must match decoded frames')
                parts = []
                for i, piece in enumerate(pieces):
                    if piece:
                        parts.append({'type': 'text', 'text': piece})
                    if i < len(frames):
                        parts.append(self._image_part(frames[i], pixels))
            representation = self._request(parts, max_pixels=pixels)
            representation['metadata'] = dict(representation['metadata'], input_modality=modality,
                                              prompt_sha256=hashlib.sha256(json.dumps(parts[0], sort_keys=True).encode()).hexdigest(),
                                              nframes=frames_count, max_pixels=pixels)
            results.append(representation)
        return results

    def encode(self, content, modality='text', **kwargs):
        records = self.encode_representations(content, modality=modality, **kwargs)
        return np.stack([record['dense'] for record in records]) if records else np.empty((0, self.dimension), dtype=np.float32)

    def encode_text(self, content, **kwargs):
        return self.encode(content, 'text', **kwargs)

    def encode_vis_query(self, content, **kwargs):
        return self.encode(content, 'vis_query', **kwargs)

    def encode_image(self, content, **kwargs):
        return self.encode(content, 'image', **kwargs)

    def encode_video(self, content, **kwargs):
        return self.encode(content, 'video', **kwargs)
