# Visual Similarity Search Engine

A CLI-based reverse image search engine. Given a query image (or even
a plain text description), it returns the most visually/semantically
similar images from an indexed dataset built on top of pretrained
CLIP embeddings and a FAISS vector index.

## Features

- **Image-to-image search** — find images visually similar to a query image
- **Text-to-image search** — find images matching a natural-language description (e.g. `"a red sports car"`), using CLIP's shared image-text embedding space
- **Precision@K evaluation** — measures retrieval quality against ground-truth class labels
- Fully CLI-driven, no GUI required
- Modular design: embedding extraction, indexing, search, and evaluation are independent components

## Technologies / Tools Used

- Python 3.10+
- PyTorch + HuggingFace `transformers` (CLIP model: `openai/clip-vit-base-patch32`)
- FAISS (`faiss-cpu`) for approximate/exact nearest-neighbour vector search
- Pillow for image I/O

## Project Structure

```
visual-similarity-search/
├── cli.py                 # CLI entrypoint (build-index / search / search-text / evaluate)
├── src/
│   ├── config.py           # central configuration
│   ├── embedder.py         # Module 1: CLIP embedding extraction
│   ├── indexer.py          # Module 2: FAISS index build/save/load
│   ├── search.py           # Module 3: similarity search / retrieval
│   ├── evaluate.py         # Module 4: Precision@K evaluation
│   └── utils.py            # logging, safe image loading
├── tests/
│   └── test_pipeline.py    # offline unit tests for retrieval logic
├── data/
│   └── images/<class_name>/*.jpg   # dataset goes here
├── index/                  # generated FAISS index + metadata (not committed)
├── docs/
│   └── report_outline.md
├── requirements.txt
└── statement.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/RohitKChoudhary/Visual-Similarity-Search-Engine.git
cd visual-similarity-search
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download the pretrained CLIP weights (~600MB) from HuggingFace — needs an internet connection once. Subsequent runs use the local cache.

### 4. Add a dataset

Organize images into class subfolders under `data/images/`:

```
data/images/
├── cat/
│   ├── cat_001.jpg
│   └── cat_002.jpg
├── dog/
│   ├── dog_001.jpg
│   └── dog_002.jpg
└── flower/
    └── flower_001.jpg
```

Any small public image dataset works (e.g. a subset of a Kaggle
animals/flowers dataset, or Kaggle's CIFAR-10 images). Aim for
5-10 classes with 20-50 images each for a meaningful demo and
evaluation.

## Running the Project

**Build the index** (extracts embeddings for every image and builds the FAISS index):

```bash
python cli.py build-index --data-dir data/images
```

**Search by image:**

```bash
python cli.py search --query data/images/cat/cat_001.jpg --top-k 5
```

**Search by text:**

```bash
python cli.py search-text --query "a fluffy orange cat" --top-k 5
```

**Evaluate retrieval quality (Precision@K):**

```bash
python cli.py evaluate --top-k 5
```

## Testing

Offline unit tests for the retrieval logic (indexing, search, evaluation)
run without downloading the CLIP model:

```bash
python -m unittest tests/test_pipeline.py -v
```

## Non-Functional Requirements

| Requirement | How it's addressed |
|---|---|
| Performance | Batched embedding extraction (configurable batch size); FAISS inner-product search is sub-millisecond at this scale |
| Scalability | `IndexFlatIP` can be swapped for an IVF/HNSW FAISS index for larger datasets without changing any other module |
| Reliability | Corrupt/unreadable images are logged and skipped, not fatal, during indexing |
| Security | All file paths are validated before use; no arbitrary code execution paths |
| Maintainability | Config values centralized in `src/config.py`; each pipeline stage is an independent, testable module |
| Logging | Structured logging (`logging` module) across every module |

## Design Diagrams

System architecture, workflow, use case, sequence, class, and data schema
diagrams are in [`docs/diagrams.md`](docs/diagrams.md) — they render
automatically when viewing that file on GitHub. For the PDF report, paste
any diagram's Mermaid code into https://mermaid.live and export as PNG/SVG.

## Design Diagrams

System architecture, workflow, use case, sequence, class, and data schema
diagrams are in [`docs/diagrams.md`](docs/diagrams.md) — they render
automatically when viewing that file on GitHub. For the PDF report, paste
any diagram's Mermaid code into https://mermaid.live and export as PNG/SVG.

## Future Enhancements

- Swap `IndexFlatIP` for `IndexIVFFlat` to scale to 100k+ images
- Add a simple FastAPI wrapper for HTTP-based querying
- Support batch/multi-image queries
