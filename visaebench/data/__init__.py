"""Data assets shipped with VISAEBench."""

from pathlib import Path

_DATA_DIR = Path(__file__).parent

IMAGENET_CLASS_INDEX_PATH = _DATA_DIR / "imagenet_class_index.json"
WORDNET_PAIRS_PATH = _DATA_DIR / "wordnet_pairs.json"
