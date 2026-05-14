"""VISAEBench: A benchmark for evaluating Sparse Autoencoders on Vision Transformers."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from visaebench.version import __version__

# Core types
from visaebench.core import (
    DIMENSIONS,
    BackboneInterface,
    EvalResults,
    MetricResult,
    SAEInterface,
)

# Backbone loading
from visaebench.backbones import CustomBackbone, list_backbones, load_backbone

# Data loaders / cache
from visaebench.data import (
    compute_or_load_norm_stats,
    default_cache_dir,
    extract_and_cache_activations,
    load_imagenet_val,
)

# SAE loading
from visaebench.hub import GenericSAE, load_sae

# Metric registry / runner
from visaebench.metrics import METRIC_REGISTRY
from visaebench.metrics.runner import MetricRunner, list_metrics

# Default OOD datasets for M6 when the user doesn't specify any.
DEFAULT_OOD_DATASETS = ("eurosat", "dtd")


def evaluate(
    sae,
    *,
    backbone=None,
    backbone_name=None,
    imagenet_path=None,
    layer=11,
    max_samples=None,
    cache_dir=None,
    activations=None,
    labels=None,
    images=None,
    mean=None,
    std=None,
    ood_datasets=None,
    ood_max_images=10_000,
    metrics="all",
    device="cuda",
    grid_size=None,
    seed=42,
    **kwargs,
):
    """Run VISAEBench evaluation on a trained SAE.

    The minimal usage is::

        import visaebench
        results = visaebench.evaluate(
            sae=my_sae,
            backbone_name="clip_vitb16",
            imagenet_path="/data/imagenet/val",   # or None for HF Hub
            device="cuda",
        )

    With nothing else specified, this:

    1. Loads the backbone (``clip_vitb16``).
    2. Loads ImageNet validation (lazy, from *imagenet_path* or HF).
    3. Extracts patch activations and caches them as fp16 shards under
       ``~/.cache/visaebench/`` (override via ``$VISAEBENCH_CACHE_DIR``).
    4. Computes per-dimension mean / scalar std for SAE input normalisation.
    5. Auto-loads the cross-model evaluator for M5 monosemanticity (e.g.
       DINOv2 when the SAE backbone is CLIP).
    6. Auto-downloads OOD datasets (EuroSAT, iNaturalist, DTD) for M6.
    7. Auto-infers ``grid_size`` from the backbone's patch count.
    8. Runs all 7 metrics and returns aggregated :class:`EvalResults`.

    Args:
        sae: Trained SAE.  Any object with ``encode(x)`` and ``decode(z)``
            satisfies :class:`SAEInterface` — no inheritance needed.
        backbone: Pre-loaded backbone instance (overrides *backbone_name*
            for extraction; *backbone_name* is still used to select the
            cross-model evaluator and OOD scheduling).
        backbone_name: Registry key (see :func:`list_backbones`).
            ``"clip_vitb16"``, ``"dinov2_vitb14"``, etc.
        imagenet_path: Directory in torchvision ``ImageFolder`` layout
            (``val/<synset>/*.JPEG``).  If ``None``, the HuggingFace
            ``imagenet-1k`` validation split is streamed instead.
        layer: Transformer block index to extract (0-indexed; default 11).
        max_samples: Optional cap on the number of ImageNet images used
            for the in-distribution metrics.  ``None`` → full 50k val.
        cache_dir: Override for the activation cache root.  Default is
            ``$VISAEBENCH_CACHE_DIR`` or ``~/.cache/visaebench``.
        activations: Pre-extracted activations (tensor ``[N, P, D]`` or
            list of shard paths).  Bypasses the auto-extraction path.
        labels: Pre-aligned integer class labels.  Required only if you
            also pass *activations* manually.
        images: Pre-aligned PIL-image sequence for M5.  Required only if
            you pass *activations* manually and want M5 to run.
        mean: Per-dimension mean ``[D]`` for SAE input normalisation.
            Auto-computed when omitted.
        std: Scalar normalisation std.  Auto-computed when omitted.
        ood_datasets: Either ``None`` (use defaults), a list of names
            (e.g. ``["eurosat", "inaturalist"]``), or a dict matching
            the legacy
            :meth:`CrossDomainGeneralization.compute` schema.
        ood_max_images: Cap per OOD dataset (default 10k).
        metrics: ``"all"`` or a sequence of metric registry keys.
        device: Torch device string.
        grid_size: Patch grid ``(H, W)`` for M1.  Auto-inferred from the
            backbone's patch count when ``None``.
        seed: Random seed.
        **kwargs: Forwarded to individual metric ``compute`` calls.

    Returns:
        :class:`EvalResults` containing one :class:`MetricResult` per
        successfully-run metric.
    """
    # ── Resolve backbone instance from name ──────────────────────────
    if backbone is None and backbone_name is not None:
        backbone = load_backbone(backbone_name, device=device)

    cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    # ── Auto-extract activations from ImageNet if not supplied ───────
    patch_count: int | None = None
    if activations is None:
        if backbone is None:
            raise ValueError(
                "evaluate() needs either pre-extracted `activations` or a "
                "`backbone`/`backbone_name` plus `imagenet_path` so it can "
                "extract them. Got none of these."
            )
        if backbone_name is None:
            raise ValueError(
                "Auto-extraction requires `backbone_name` so the cache key "
                "and cross-model evaluator can be resolved."
            )

        source_tag = (
            f"imagenet_{Path(imagenet_path).resolve().name}"
            if imagenet_path is not None
            else "imagenet_hf"
        )

        img_seq, lbl_arr = load_imagenet_val(
            imagenet_path=imagenet_path,
            max_samples=max_samples,
            seed=seed,
        )
        # Default labels/images to what we just loaded — but only if the
        # caller didn't override either.
        if labels is None:
            labels = lbl_arr
        if images is None:
            images = img_seq

        shard_paths, patch_count = extract_and_cache_activations(
            backbone=backbone,
            backbone_name=backbone_name,
            images=img_seq,
            layer=layer,
            source_tag=source_tag,
            cache_dir=cache_root,
        )
        activations = shard_paths

        # Auto-compute normalisation stats (cached next to the shards).
        if mean is None or std is None:
            stats_path = shard_paths[0].parent / "norm_stats.pt"
            mean, std = compute_or_load_norm_stats(
                shard_paths, cache_file=stats_path,
            )

    # ── Auto-infer grid_size from patch count ────────────────────────
    if grid_size is None:
        if patch_count is None:
            patch_count = _peek_patch_count(activations)
        grid_size = _infer_grid_size(patch_count)

    # ── Auto-build OOD datasets for M6 ───────────────────────────────
    if ood_datasets is None:
        ood_datasets = list(DEFAULT_OOD_DATASETS)
    if isinstance(ood_datasets, (list, tuple)) and ood_datasets and isinstance(
        ood_datasets[0], str
    ):
        ood_datasets = _build_ood_entries(
            ood_datasets, max_images=ood_max_images, seed=seed,
        )

    # ── Run metrics ──────────────────────────────────────────────────
    runner = MetricRunner(metrics=metrics, device=device)
    results = runner.run(
        sae,
        backbone=backbone,
        backbone_name=backbone_name,
        activations=activations,
        labels=labels,
        images=images,
        mean=mean,
        std=std,
        grid_size=grid_size,
        seed=seed,
        ood_datasets=ood_datasets,
        **kwargs,
    )

    print()
    print(results.summary())
    return results


# ======================================================================
# Helpers
# ======================================================================


def _peek_patch_count(activations) -> int:
    """Inspect the first shard / batch and return the P dimension."""
    if isinstance(activations, torch.Tensor):
        return int(activations.shape[1]) if activations.ndim == 3 else 1
    # Treat as a sequence of shard paths.
    first = torch.load(activations[0], map_location="cpu", weights_only=True)
    return int(first.shape[1]) if first.ndim == 3 else 1


def _infer_grid_size(patch_count: int) -> tuple[int, int]:
    """Map a flat patch count to ``(H, W)`` assuming a square grid."""
    side = int(round(math.sqrt(patch_count)))
    if side * side != patch_count:
        raise ValueError(
            f"Cannot infer a square grid from patch count {patch_count}. "
            f"Pass grid_size=(H, W) explicitly."
        )
    return (side, side)


def _build_ood_entries(
    names: Sequence[str], *, max_images: int, seed: int,
) -> dict[str, dict]:
    """Lazy-download a list of OOD dataset names and return entries
    suitable for :meth:`CrossDomainGeneralization.compute`."""
    from visaebench.metrics.concept_detection.cross_domain import load_ood_dataset

    entries: dict[str, dict] = {}
    for name in names:
        try:
            imgs, lbls = load_ood_dataset(
                name, max_images=max_images, seed=seed,
            )
        except Exception as e:
            warnings.warn(
                f"OOD dataset {name} failed to load: {e}. "
                f"M6 will run on remaining datasets.",
                RuntimeWarning,
            )
            continue
        entries[name] = {"images": imgs, "labels": lbls}
    return entries


__all__ = [
    "__version__",
    # Core types
    "DIMENSIONS",
    "BackboneInterface",
    "EvalResults",
    "MetricResult",
    "SAEInterface",
    # Backbone
    "CustomBackbone",
    "list_backbones",
    "load_backbone",
    # Data helpers
    "compute_or_load_norm_stats",
    "default_cache_dir",
    "extract_and_cache_activations",
    "load_imagenet_val",
    # Metrics
    "METRIC_REGISTRY",
    "MetricRunner",
    "list_metrics",
    # SAE loading
    "GenericSAE",
    "load_sae",
    # Top-level API
    "DEFAULT_OOD_DATASETS",
    "evaluate",
]
