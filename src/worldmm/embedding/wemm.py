"""Dense-only local WeMM adapter; model dependencies load only on first use.

Run in the WeMM environment (Transformers 5.2.0, SentenceTransformers >=5.7.0,
qwen-vl-utils[decord] 0.0.14), independently of legacy embedding dependencies.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image


class WeMMEmbeddingModel:
    def __init__(self, model_name='/myworkspace/models/tencent/WeMM-Embedding-4B',
                 device='cuda', dimensions=2560, max_pixels=262144, nframes=16,
                 model_config=None):
        config = dict(model_config or {})
        self.model_name = config.get('model_path', model_name)
        self.device = device
        self.dimension = int(config.get('embedding_dim', dimensions))
        self.max_pixels = int(config.get('max_image_pixels', max_pixels))
        self.nframes = int(config.get('nframes', nframes))
        if min(self.dimension, self.max_pixels, self.nframes) <= 0:
            raise ValueError('WeMM dimensions, pixels and frames must be positive')
        self._model = None

    @property
    def fingerprint(self):
        return {'backend': 'wemm4b', 'model': self.model_name,
                'weights_revision': os.environ.get('WORLDMM_WEMM_REVISION', 'UNVERIFIED'),
                'dimensions': self.dimension, 'normalization': 'l2-float32',
                'text_roles': 'encode_query/encode_document', 'visual_input': 'chat-frames-timestamps-v1',
                'max_pixels': self.max_pixels, 'nframes': self.nframes}

    @property
    def index_identity(self):
        return self.fingerprint

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)
        return self._model

    def load_model(self, model_type=None):
        if model_type not in (None, 'text', 'vision'):
            raise ValueError('Unknown model type')
        self._load_model()

    def _image(self, value):
        if isinstance(value, dict):
            value = value['image']
        if isinstance(value, (str, Path)):
            with Image.open(value) as opened:
                image = opened.convert('RGB')
        elif isinstance(value, Image.Image):
            image = value.convert('RGB')
        else:
            raise TypeError('WeMM images require paths or PIL images')
        if image.width * image.height > self.max_pixels:
            scale = (self.max_pixels / (image.width * image.height)) ** .5
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        return image

    def _video_document(self, value, nframes):
        from .freeret_api import FreeRetAPI
        frames, timestamps = FreeRetAPI._video_frames(self, value, nframes=nframes, with_timestamps=True)
        content = []
        for frame, timestamp in zip(frames, timestamps):
            content.extend([{'type': 'text', 'text': f'{timestamp:.3f}s'},
                            {'type': 'image', 'image': self._image(frame)}])
        return [{'role': 'user', 'content': content}]

    def encode(self, content, modality='text', *, role='passage', batch_size=16, nframes=None, **kwargs):
        if kwargs:
            raise TypeError(f'Unsupported WeMM encode options: {sorted(kwargs)}')
        if role not in ('passage', 'query'):
            raise ValueError('WeMM role must be passage or query')
        items = content if isinstance(content, list) else [content]
        if not items:
            return np.empty((0, self.dimension), dtype=np.float32)
        if modality in ('text', 'vis_query'):
            if any(not isinstance(item, str) for item in items):
                raise TypeError('WeMM text inputs must be strings')
            documents = items
        elif modality == 'image':
            documents = [{'image': self._image(item)} for item in items]
        elif modality == 'video':
            frame_count = self.nframes if nframes is None else nframes
            if type(frame_count) is not int or frame_count <= 0:
                raise ValueError('nframes must be a positive integer')
            documents = [self._video_document(item, frame_count) for item in items]
        else:
            raise ValueError(f'Unknown WeMM modality: {modality}')
        model = self._load_model()
        encode = model.encode_query if role == 'query' or modality == 'vis_query' else model.encode_document
        dense = np.asarray(encode(documents, batch_size=batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True), dtype=np.float32)
        if dense.shape != (len(items), self.dimension):
            raise ValueError('WeMM returned invalid embedding dimensions or batch size')
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        if not np.isfinite(dense).all() or not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError('WeMM requires finite nonzero dense embeddings')
        return dense / norms

    def encode_text(self, content, **kwargs):
        return self.encode(content, 'text', **kwargs)

    def encode_vis_query(self, content, **kwargs):
        return self.encode(content, 'vis_query', **kwargs)

    def encode_image(self, content, **kwargs):
        return self.encode(content, 'image', **kwargs)

    def encode_video(self, content, **kwargs):
        return self.encode(content, 'video', **kwargs)

    def encode_representations(self, *args, **kwargs):
        raise ValueError('WeMM is dense-only and does not provide FreeRet sparse tokens')
