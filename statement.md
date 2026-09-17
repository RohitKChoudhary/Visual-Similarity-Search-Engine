# Problem Statement

Manually searching for visually similar images in a large, unlabeled
or loosely-labeled image collection is slow and doesn't scale — text
tags are often missing, inconsistent, or simply don't capture visual
similarity (pose, color, texture, composition). A system that lets a
user query "find me images like this one" — or even describe what
they're looking for in plain language — using the actual visual (and
semantic) content of the images solves this directly.

## Scope of the Project

This project implements a Visual Similarity Search Engine that:

1. Extracts dense semantic embeddings for a directory of images using
   a pretrained CLIP vision-language model (no training required).
2. Indexes those embeddings using FAISS for fast nearest-neighbour
   retrieval.
3. Supports two query modes:
   - **Image-to-image**: supply a query image, retrieve the most
     visually/semantically similar images in the dataset.
   - **Text-to-image**: supply a natural-language description,
     retrieve images matching that description (enabled by CLIP's
     shared image-text embedding space).
4. Evaluates retrieval quality quantitatively using Precision@K
   against ground-truth class labels (derived from dataset folder
   structure).

The scope is intentionally focused on retrieval, not classification
or object detection — the core computer vision skill being
demonstrated is feature extraction via a pretrained CNN/vision
transformer and its application to a real retrieval task, backed by
measurable evaluation.

## Target Users

- Anyone managing a medium-sized personal or organizational image
  collection (photographers, e-commerce catalogs, digital asset
  libraries) who wants to search by visual content rather than
  manually maintained tags.
- As an academic project, it also serves as a demonstrable, testable
  application of transfer learning and vector similarity search.

## High-Level Features

- CLI-driven image-to-image similarity search
- CLI-driven text-to-image similarity search
- Fast vector search via FAISS (constant-time-ish lookups even as the
  dataset grows, especially once swapped to an IVF index)
- Quantitative evaluation via Precision@K
- Modular, testable pipeline (embedding, indexing, search, evaluation
  are independent components with their own unit tests)
