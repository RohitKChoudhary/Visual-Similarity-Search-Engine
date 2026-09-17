"""
Module 1: Embedding Extraction.

Responsible for turning images (and, thanks to CLIP being a joint
image-text model, also text queries) into fixed-length dense vectors
that live in the same embedding space. This is what makes the
retrieval module possible.

Design choice: use a pretrained CLIP model rather than training a CNN
from scratch. Training a classifier from scratch on a small custom
dataset would overfit badly and eats time we don't have; CLIP's
embeddings are already well-separated for semantic similarity out of
the box (zero-shot), which is the standard, defensible approach for a
retrieval system like this.
"""

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config import CLIP_MODEL_NAME
from src.utils import get_logger

logger = get_logger(__name__)


class ClipEmbedder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        logger.info("Model loaded.")

    @staticmethod
    def _extract_embeds(output, embeds_attr: str):
        """
        transformers versions differ in what get_image_features/get_text_features
        return: older versions return a plain tensor, newer versions (>=4.5x, the
        BaseModelOutputWithPooling change) return a structured output object.
        Handle both so this works regardless of installed version.
        """
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, embeds_attr):
            return getattr(output, embeds_attr)
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        raise AttributeError(
            f"Unexpected output type from CLIP model: {type(output)}. "
            f"Could not find '{embeds_attr}' or 'pooler_output'."
        )

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Embed a list of PIL images. Returns (N, D) L2-normalized float32 array."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            output = self.model.get_image_features(**inputs)
            features = self._extract_embeds(output, "image_embeds")
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())
        return np.vstack(all_embeddings).astype("float32")

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text query. Returns (1, D) L2-normalized float32 array."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        output = self.model.get_text_features(**inputs)
        features = self._extract_embeds(output, "text_embeds")
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")
