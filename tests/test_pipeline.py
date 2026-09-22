"""
Lightweight tests for the indexing/search/evaluation logic using
synthetic embeddings — does NOT require downloading CLIP weights,
so it runs fast and offline. This tests the retrieval pipeline
correctness independent of the embedding model.

Run with: python -m pytest tests/test_pipeline.py -v
(or: python tests/test_pipeline.py)
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import FaissIndexer
from src.search import search
from src.evaluate import precision_at_k


class TestPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.paths = []
        embs = []
        for cls in ["cat", "dog", "bird"]:
            base = np.random.randn(8).astype("float32")
            for i in range(4):
                v = base + np.random.randn(8).astype("float32") * 0.05
                v = v / np.linalg.norm(v)
                embs.append(v)
                self.paths.append(f"data/images/{cls}/{cls}_{i}.jpg")
        self.embeddings = np.vstack(embs).astype("float32")

        self.indexer = FaissIndexer(embedding_dim=8)
        self.indexer.build(self.embeddings, self.paths)

    def test_index_build_count(self):
        self.assertEqual(self.indexer.index.ntotal, 12)

    def test_self_search_returns_self_as_top_hit(self):
        results = search(self.indexer, self.embeddings[0:1], top_k=1)
        self.assertEqual(results[0][0], self.paths[0])
        self.assertAlmostEqual(results[0][1], 1.0, places=3)

    def test_top_k_capped_to_index_size(self):
        results = search(self.indexer, self.embeddings[0:1], top_k=100)
        self.assertEqual(len(results), 12)

    def test_precision_at_k_on_separable_clusters(self):
        mean_p, per_query = precision_at_k(self.indexer, self.embeddings, k=3)
        self.assertEqual(len(per_query), 12)
        self.assertGreater(mean_p, 0.9)  # tight clusters -> near-perfect precision

    def test_save_and_load_round_trip(self):
        self.indexer.save("/tmp/_test_faiss.index", "/tmp/_test_metadata.json")
        loaded = FaissIndexer.load("/tmp/_test_faiss.index", "/tmp/_test_metadata.json")
        self.assertEqual(loaded.index.ntotal, self.indexer.index.ntotal)
        self.assertEqual(loaded.image_paths, self.indexer.image_paths)


if __name__ == "__main__":
    unittest.main()
