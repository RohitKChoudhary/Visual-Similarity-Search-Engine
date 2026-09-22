"""
Module 2: Indexing.

Wraps FAISS index construction, persistence, and loading. Kept
separate from search.py (querying) so indexing and retrieval can
scale/be optimized independently — e.g. swapping IndexFlatIP for an
IVF index as the dataset grows is a one-line change here that doesn't
touch anything else (this is the "scalability" non-functional
requirement in practice).
"""

import json
import os
from typing import List

import faiss
import numpy as np

from src.config import EMBEDDING_DIM
from src.utils import get_logger

logger = get_logger(__name__)


class FaissIndexer:
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        # Inner product on L2-normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.image_paths: List[str] = []

    def build(self, embeddings: np.ndarray, image_paths: List[str]) -> None:
        if embeddings.shape[0] != len(image_paths):
            raise ValueError("Embeddings count must match image_paths count")
        self.index.add(embeddings)
        self.image_paths = image_paths
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors.")

    def save(self, index_path: str, metadata_path: str) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w") as f:
            json.dump({"image_paths": self.image_paths}, f)
        logger.info(f"Saved index -> {index_path}, metadata -> {metadata_path}")

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FaissIndexer":
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                "Index or metadata not found. Run `build-index` first."
            )
        index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        obj = cls(embedding_dim=index.d)
        obj.index = index
        obj.image_paths = metadata["image_paths"]
        logger.info(f"Loaded FAISS index with {obj.index.ntotal} vectors.")
        return obj
