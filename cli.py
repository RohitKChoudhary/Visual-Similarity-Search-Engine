#!/usr/bin/env python3
"""
Visual Similarity Search Engine — CLI entrypoint.

Usage:
    python cli.py build-index --data-dir data/images
    python cli.py search --query path/to/query.jpg --top-k 5
    python cli.py search-text --query "a red sports car" --top-k 5
    python cli.py evaluate --top-k 5

Run `python cli.py <command> --help` for full options.
"""

import argparse
import sys

from src.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_TOP_K,
)
from src.embedder import ClipEmbedder
from src.evaluate import precision_at_k
from src.indexer import FaissIndexer
from src.search import search as run_search
from src.utils import get_logger, list_images, safe_load_image

logger = get_logger("cli")


def cmd_build_index(args):
    image_paths = list_images(args.data_dir)
    if not image_paths:
        logger.error(f"No supported images found in {args.data_dir}")
        sys.exit(1)
    logger.info(f"Found {len(image_paths)} candidate images.")

    embedder = ClipEmbedder()

    valid_paths, valid_images = [], []
    for p in image_paths:
        img = safe_load_image(p)
        if img is not None:
            valid_paths.append(p)
            valid_images.append(img)

    if not valid_images:
        logger.error("No images could be loaded successfully.")
        sys.exit(1)

    embeddings = embedder.embed_images(valid_images, batch_size=args.batch_size)

    indexer = FaissIndexer(embedding_dim=embeddings.shape[1])
    indexer.build(embeddings, valid_paths)
    indexer.save(args.index_path, args.metadata_path)
    logger.info("Index build complete.")


def cmd_search(args):
    embedder = ClipEmbedder()
    indexer = FaissIndexer.load(args.index_path, args.metadata_path)

    query_img = safe_load_image(args.query)
    if query_img is None:
        logger.error(f"Could not load query image: {args.query}")
        sys.exit(1)

    query_embedding = embedder.embed_images([query_img], batch_size=1)
    results = run_search(indexer, query_embedding, top_k=args.top_k)

    print(f"\nTop {len(results)} results for '{args.query}':")
    for rank, (path, score) in enumerate(results, start=1):
        print(f"  {rank}. {path}  (similarity: {score:.4f})")


def cmd_search_text(args):
    embedder = ClipEmbedder()
    indexer = FaissIndexer.load(args.index_path, args.metadata_path)

    query_embedding = embedder.embed_text(args.query)
    results = run_search(indexer, query_embedding, top_k=args.top_k)

    print(f"\nTop {len(results)} results for text query \"{args.query}\":")
    for rank, (path, score) in enumerate(results, start=1):
        print(f"  {rank}. {path}  (similarity: {score:.4f})")


def cmd_evaluate(args):
    embedder = ClipEmbedder()
    indexer = FaissIndexer.load(args.index_path, args.metadata_path)

    images = [safe_load_image(p) for p in indexer.image_paths]
    valid = [(p, img) for p, img in zip(indexer.image_paths, images) if img is not None]
    if len(valid) != len(indexer.image_paths):
        logger.warning("Some indexed images could not be reloaded for evaluation.")

    paths, imgs = zip(*valid)
    embeddings = embedder.embed_images(list(imgs), batch_size=args.batch_size)

    mean_p, _ = precision_at_k(indexer, embeddings, k=args.top_k)
    print(f"\nPrecision@{args.top_k}: {mean_p:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Visual Similarity Search Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="Extract embeddings and build the FAISS index")
    p_build.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p_build.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    p_build.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    p_build.add_argument("--batch-size", type=int, default=16)
    p_build.set_defaults(func=cmd_build_index)

    p_search = sub.add_parser("search", help="Find images similar to a query image")
    p_search.add_argument("--query", required=True, help="Path to query image")
    p_search.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_search.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    p_search.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    p_search.set_defaults(func=cmd_search)

    p_search_text = sub.add_parser("search-text", help="Find images matching a text description")
    p_search_text.add_argument("--query", required=True, help="Text description")
    p_search_text.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_search_text.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    p_search_text.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    p_search_text.set_defaults(func=cmd_search_text)

    p_eval = sub.add_parser("evaluate", help="Compute Precision@K over the indexed dataset")
    p_eval.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_eval.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    p_eval.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
