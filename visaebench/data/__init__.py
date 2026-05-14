"""Data assets and loading helpers for VISAEBench."""

from pathlib import Path

from visaebench.data.cache import (
    compute_or_load_norm_stats,
    default_cache_dir,
    extract_and_cache_activations,
)
from visaebench.data.loaders import load_imagenet_val

_DATA_DIR = Path(__file__).parent

IMAGENET_CLASS_INDEX_PATH = _DATA_DIR / "imagenet_class_index.json"
WORDNET_PAIRS_PATH = _DATA_DIR / "wordnet_pairs.json"

__all__ = [
    "IMAGENET_CLASS_INDEX_PATH",
    "WORDNET_PAIRS_PATH",
    "compute_or_load_norm_stats",
    "default_cache_dir",
    "extract_and_cache_activations",
    "load_imagenet_val",
]
