"""
Module 3: Search / Retrieval.

Takes a query embedding (from either an image or a text prompt) and
returns the top-K most similar images from the index, with scores.
"""

from typing import List, Tuple

import numpy as np

from src.indexer import FaissIndexer
from src.utils import get_logger

logger = get_logger(__name__)


def search(indexer: FaissIndexer, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    query_embedding: (1, D) L2-normalized vector.
    Returns list of (image_path, similarity_score) sorted by descending similarity.
    """
    top_k = min(top_k, indexer.index.ntotal)
    if top_k == 0:
        return []

    scores, indices = indexer.index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((indexer.image_paths[idx], float(score)))
    return results
