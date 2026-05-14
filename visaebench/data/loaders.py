"""ImageNet validation loaders for VISAEBench.

Provides ``load_imagenet_val`` which returns aligned ``(images, labels)``
from either a local ``torchvision.datasets.ImageFolder`` layout or the
HuggingFace ``imagenet-1k`` hub, with lazy per-index access so the full
50k validation set does not need to live in RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class _LazyImages:
    """Lazy indexable PIL image sequence.

    ``__getitem__(i)`` opens the i-th image on demand; ``len(self)``
    returns the (subsampled) length.  This is what M5 monosemanticity
    expects — a sequence-like ``images`` argument that can be sliced
    by integer index.
    """

    __slots__ = ("_loader", "_indices")

    def __init__(self, loader, indices: Sequence[int]):
        self._loader = loader
        self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._loader(j) for j in self._indices[i]]
        return self._loader(self._indices[i])

    def __iter__(self) -> Iterable:
        for j in self._indices:
            yield self._loader(j)


def load_imagenet_val(
    imagenet_path: str | None = None,
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[_LazyImages, np.ndarray]:
    """Load ImageNet validation images and labels.

    Auto-detects the source:

    * If *imagenet_path* points to a directory, treat it as a torchvision
      ``ImageFolder`` (synset-name folders, e.g.
      ``val/n01440764/*.JPEG``).
    * If *imagenet_path* is ``None``, stream the validation split of
      ``imagenet-1k`` from the HuggingFace Hub (requires gated dataset
      access on the host's HF token).

    Args:
        imagenet_path: Filesystem path to ImageNet val, or ``None`` for HF.
        max_samples: If set, randomly subsample this many images
            (deterministically, seeded by *seed*).  Default ``None`` →
            return the full validation set.
        seed: Random seed for subsampling.

    Returns:
        ``(images, labels)`` where *images* is a lazy indexable
        :class:`_LazyImages` (returns a PIL image on ``images[i]``),
        and *labels* is a length-N ``np.int64`` array aligned with it.
        Labels use the canonical 0–999 ImageNet ordering (alphabetical
        synset id), which matches ``imagenet_class_index.json``.
    """
    if imagenet_path is not None:
        return _load_from_imagefolder(Path(imagenet_path), max_samples, seed)
    return _load_from_huggingface(max_samples, seed)


# ----------------------------------------------------------------------
# Local ImageFolder
# ----------------------------------------------------------------------


def _load_from_imagefolder(
    path: Path, max_samples: int | None, seed: int,
) -> tuple[_LazyImages, np.ndarray]:
    from torchvision.datasets import ImageFolder

    if not path.is_dir():
        raise FileNotFoundError(
            f"imagenet_path {str(path)!r} is not a directory. Expected a "
            f"torchvision ImageFolder layout (val/<synset>/*.JPEG)."
        )

    # ImageFolder enumerates classes alphabetically by folder name, which
    # for ImageNet synsets matches the canonical 0-999 ordering used in
    # imagenet_class_index.json.
    ds = ImageFolder(str(path))
    labels_full = np.array(ds.targets, dtype=np.int64)

    indices = _subsample_indices(len(ds), max_samples, seed)
    labels = labels_full[indices]

    def _load(i: int):
        # ImageFolder.__getitem__ returns (PIL.Image, label).  We only
        # want the image and re-attach labels via *labels* above.
        return ds[i][0].convert("RGB")

    return _LazyImages(_load, indices), labels


# ----------------------------------------------------------------------
# HuggingFace Hub fallback
# ----------------------------------------------------------------------


def _load_from_huggingface(
    max_samples: int | None, seed: int,
) -> tuple[_LazyImages, np.ndarray]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Loading ImageNet from HuggingFace requires the `datasets` "
            "package. Install with `pip install datasets`, or pass "
            "`imagenet_path=<local ImageFolder>` instead."
        ) from e

    ds = load_dataset("imagenet-1k", split="validation")
    label_col = "label" if "label" in ds.column_names else "labels"

    # `ds["label"]` is a fast bulk fetch for the integer label column;
    # image bytes are still loaded lazily on per-row access.
    labels_full = np.array(ds[label_col], dtype=np.int64)

    indices = _subsample_indices(len(ds), max_samples, seed)
    labels = labels_full[indices]

    def _load(i: int):
        return ds[int(i)]["image"].convert("RGB")

    return _LazyImages(_load, indices), labels


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _subsample_indices(
    total: int, max_n: int | None, seed: int,
) -> np.ndarray:
    if max_n is None or total <= max_n:
        return np.arange(total, dtype=np.int64)
    rng = np.random.RandomState(seed)
    idx = rng.choice(total, size=max_n, replace=False)
    idx.sort()
    return idx
