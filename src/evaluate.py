"""
Module 4: Evaluation.

Measures retrieval quality with Precision@K. Assumes the dataset is
organized as data/images/<class_name>/<image>.jpg — the parent folder
name is treated as the ground-truth label. For each image in the
index, we query the index with that image itself (excluding the
image against itself) and check what fraction of the top-K results
share the same label.

This is what turns the project from "a demo that looks like it works"
into something with a measurable, defensible result for the report's
"Testing Approach" and "Results" sections.
"""

import os
from typing import List, Tuple

import numpy as np

from src.indexer import FaissIndexer
from src.utils import get_logger

logger = get_logger(__name__)


def _label_of(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def precision_at_k(indexer: FaissIndexer, embeddings: np.ndarray, k: int = 5) -> Tuple[float, List[float]]:
    """
    Returns (mean_precision_at_k, per_query_precisions).
    embeddings must be in the same order as indexer.image_paths.
    """
    n = len(indexer.image_paths)
    per_query = []

    for i in range(n):
        query_vec = embeddings[i:i + 1]
        # Search for k+1 to account for the image matching itself, then drop self-match.
        scores, indices = indexer.index.search(query_vec, k + 1)
        true_label = _label_of(indexer.image_paths[i])

        hits = 0
        counted = 0
        for idx in indices[0]:
            if idx == i or idx == -1:
                continue
            if counted >= k:
                break
            if _label_of(indexer.image_paths[idx]) == true_label:
                hits += 1
            counted += 1

        if counted > 0:
            per_query.append(hits / counted)

    mean_p = float(np.mean(per_query)) if per_query else 0.0
    logger.info(f"Precision@{k}: {mean_p:.4f} over {len(per_query)} queries")
    return mean_p, per_query
