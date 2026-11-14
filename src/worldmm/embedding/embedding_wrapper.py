import os
import numpy as np
from typing import Union, List, Dict, Any, Optional
from PIL import Image


class EmbeddingModel:
    """Universal embedding wrapper that routes different modalities to appropriate models"""
    
    def __init__(self, 
                text_model_name: str = "/myworkspace/mymodels/Qwen3-Embedding-4B",
                vis_model_name: str = "/myworkspace/mymodels/VLM2Vec",
                device: str = "cuda",
                backend: Optional[str] = None,
                **freeret_options):
        """
        Initialize embedding models for different modalities
        
        Args:
            text_model_name: Model name for text embeddings (defaults to Qwen3-Embedding-4B)
            vis_model_name: Model name for visual embeddings (defaults to VLM2Vec V2.0)
            device: Device to run models on
        """
        self._explicit_backend = backend is not None or "WORLDMM_EMBEDDING_BACKEND" in os.environ
        self.backend = backend or os.environ.get("WORLDMM_EMBEDDING_BACKEND", "original")
        if self.backend not in ("original", "freeret", "wemm4b"):
            raise ValueError(f"Unknown embedding backend: {self.backend}")
        self._freeret = None
        if self.backend == "freeret":
            from .freeret_api import FreeRetAPI
            if "model_config" not in freeret_options:
                from worldmm.retrieval_config import load_videospy_config
                config = load_videospy_config()
                search = config.get("tools", {}).get("semantic_search", {})
                freeret_options["model_config"] = search.get("model", {})
                method = search.get(search.get("index_method", "single"), {})
                templates = {}
                for source, target in (("query_template", "vis_query"), ("clip_template", "video")):
                    if source in method:
                        templates[target] = method[source]
                freeret_options.setdefault("prompt_templates", templates)
            self._freeret = FreeRetAPI(**freeret_options)
            self.text_model_name = self.vis_model_name = self._freeret.model_name
            self.device = device
            return
        if self.backend == "wemm4b":
            from .wemm import WeMMEmbeddingModel
            if "model_config" not in freeret_options:
                from worldmm.retrieval_config import load_videospy_config
                config = load_videospy_config()
                freeret_options["model_config"] = config.get("tools", {}).get("semantic_search", {}).get("model", {})
            self._freeret = WeMMEmbeddingModel(
                model_name=os.environ.get("WORLDMM_WEMM_MODEL", "/myworkspace/models/tencent/WeMM-Embedding-4B"),
                device=os.environ.get("WORLDMM_EMBEDDING_DEVICE", device), **freeret_options)
            self.text_model_name = self.vis_model_name = self._freeret.model_name
            self.device = device
            return
        if freeret_options:
            raise TypeError("FreeRet options require the freeret backend")
        self.device = os.environ.get("WORLDMM_EMBEDDING_DEVICE", device)
        # The vision encoder (VLM2Vec) is only needed per retrieval query, so it
        # can live on a separate (e.g. CPU) device when the GPU is shared with
        # the LLM server and the text embedding model on a single card.
        self.vis_device = os.environ.get("WORLDMM_VIS_EMBEDDING_DEVICE", self.device)
        
        # Initialize models lazily
        self._text_model = None
        self._vis_model = None
        self.text_model_name = os.environ.get("WORLDMM_TEXT_EMBEDDING_MODEL", text_model_name)
        self.vis_model_name = os.environ.get("WORLDMM_VISUAL_EMBEDDING_MODEL", vis_model_name)

        # Eagerly load models
        # self._text_model = self.text_model
        # self._vis_model = self.vis_model

    @property
    def text_model(self):
        """Lazy loading of text model"""
        if self._freeret is not None:
            return self._freeret
        if self._text_model is None:
            from .qwen3_embedding import Qwen3EmbeddingModel as TextEmbeddingModel
            self._text_model = TextEmbeddingModel(
                model_name=self.text_model_name,
                device=self.device
            )
        return self._text_model
    
    @property
    def vis_model(self):
        """Lazy loading of visual model"""
        if self._freeret is not None:
            return self._freeret
        if self._vis_model is None:
            from .vlm2vecv2 import VLM2VecV2EmbeddingModel as VisEmbeddingModel
            self._vis_model = VisEmbeddingModel(
                model_name=self.vis_model_name,
                device=self.vis_device
            )
        return self._vis_model

    def load_model(self, model_type: Optional[str] = None):
        """Load embedding models based on specified type"""
        if self.backend == "wemm4b":
            return self._freeret.load_model(model_type)
        if model_type is None:
            _ = self.text_model
            _ = self.vis_model
        elif model_type == "text":
            _ = self.text_model
        elif model_type == "vision":
            _ = self.vis_model
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from None, 'text', or 'vision'")

    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode text using Qwen3 model"""
        if self._freeret is not None:
            return self._freeret.encode(texts, modality="text", **kwargs)
        return self.text_model.encode_text(texts, **kwargs)

    def encode_text_query(self, texts, **kwargs) -> np.ndarray:
        if self._freeret is not None:
            return self._freeret.encode(texts, modality="text", role="query", **kwargs)
        return self.encode_text(texts, **kwargs)

    def encode_vis_query(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode visual query using VLM2VecV2 model"""
        if self._freeret is not None:
            return self._freeret.encode(texts, modality="vis_query", **kwargs)
        return self.vis_model.encode_text(texts, **kwargs)

    def encode_image(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> np.ndarray:
        """Encode images using VLM2VecV2 model"""
        if self._freeret is not None:
            return self._freeret.encode(images, modality="image", **kwargs)
        return self.vis_model.encode_image(images, **kwargs)
    
    def encode_video(self, video_paths: Union[str, Dict[str, Any], List[str], List[Dict[str, Any]]], **kwargs) -> np.ndarray:
        """Encode videos using VLM2VecV2 model"""
        if self._freeret is not None:
            if "num_frames" in kwargs:
                kwargs.setdefault("nframes", kwargs.pop("num_frames"))
            return self._freeret.encode(video_paths, modality="video", **kwargs)
        return self.vis_model.encode_video(video_paths, **kwargs)
    
    @property
    def index_identity(self):
        # Legacy callers keep their existing directory layout. Explicit experiments
        # opt into identity-checked, backend-specific persistence.
        if self._freeret is not None:
            return self._freeret.index_identity
        return self.fingerprint if self._explicit_backend else None

    @property
    def fingerprint(self):
        if self._freeret is not None:
            return self._freeret.fingerprint
        return {"backend": "original", "text_model": self.text_model_name,
                "visual_model": self.vis_model_name,
                "visual_base_model": os.environ.get("WORLDMM_VLM_BACKBONE_MODEL", "/myworkspace/mymodels/Qwen2-VL-2B-Instruct")}

    def encode_representations(self, content, modality="text", **kwargs):
        if self._freeret is None:
            raise ValueError("dense+sparse representations require freeret backend")
        if "num_frames" in kwargs:
            kwargs.setdefault("nframes", kwargs.pop("num_frames"))
        return self._freeret.encode_representations(content, modality=modality, **kwargs)

    def encode(self, content: Union[str, List[str], Image.Image, List[Image.Image]], 
               modality: str = "text", **kwargs) -> np.ndarray:
        """
        Universal encode method that routes to appropriate model based on modality
        
        Args:
            content: Content to encode (text, images, or video paths)
            modality: Type of content ('text', 'image', 'video', 'vis_query')
            **kwargs: Additional arguments for specific encoders
        
        Returns:
            numpy array of embeddings
        """
        if modality == "text":
            return self.encode_text(content, **kwargs)  # type: ignore
        elif modality == "image":
            return self.encode_image(content, **kwargs)  # type: ignore
        elif modality == "video":
            return self.encode_video(content, **kwargs)  # type: ignore
        elif modality == "vis_query":
            return self.encode_vis_query(content, **kwargs)  # type: ignore
        else:
            raise ValueError(f"Unsupported modality: {modality}. Choose from 'text', 'image', 'video', 'vis_query'")
