"""
Shared utility functions: logging setup and safe image loading.

Centralizing error handling here means every module gets consistent
behavior when it hits a corrupt file, instead of each module
reinventing its own try/except (this is the "reliability" and
"error handling strategy" non-functional requirement in practice).
"""

import logging
import os
from typing import List, Optional

from PIL import Image, UnidentifiedImageError

from src.config import LOG_FORMAT, LOG_LEVEL, SUPPORTED_EXTENSIONS


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL))
    return logger


def list_images(data_dir: str) -> List[str]:
    """Return sorted absolute paths of all supported images under data_dir."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXTENSIONS):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def safe_load_image(path: str) -> Optional[Image.Image]:
    """
    Load an image, converting to RGB. Returns None (instead of raising)
    on a corrupt/unreadable file so batch jobs can skip and continue
    rather than crashing the whole indexing run.
    """
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger = get_logger(__name__)
        logger.warning(f"Skipping unreadable image {path}: {e}")
        return None
