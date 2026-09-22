"""
Central configuration for the Visual Similarity Search Engine.

Keeping every tunable value here (instead of scattered magic numbers)
is what makes the codebase maintainable — one of the non-functional
requirements for this project.
"""

import os

# --- Model ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512  # output dim of clip-vit-base-patch32

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "images")
DEFAULT_INDEX_DIR = os.path.join(BASE_DIR, "index")
DEFAULT_INDEX_PATH = os.path.join(DEFAULT_INDEX_DIR, "faiss.index")
DEFAULT_METADATA_PATH = os.path.join(DEFAULT_INDEX_DIR, "metadata.json")

# --- Search ---
DEFAULT_TOP_K = 5

# --- Supported image formats ---
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
