"""VISAEBench: A benchmark for evaluating Sparse Autoencoders on Vision Transformers."""

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

# SAE loading
from visaebench.hub import GenericSAE, load_sae

# Metric registry
from visaebench.metrics import METRIC_REGISTRY
from visaebench.metrics.runner import MetricRunner, list_metrics


def evaluate(
    sae,
    *,
    backbone=None,
    backbone_name=None,
    activations=None,
    labels=None,
    images=None,
    mean=None,
    std=None,
    metrics="all",
    device="cuda",
    grid_size=(14, 14),
    seed=42,
    **kwargs,
):
    """Run VISAEBench evaluation on a trained SAE.

    This is the main entry point.  For a minimal evaluation (just FVU)::

        import visaebench
        results = visaebench.evaluate(
            sae=my_sae,
            activations=shard_paths,
            mean=mean, std=std,
            metrics=["fvu"],
            device="cuda",
        )
        results.summary()

    For the full benchmark with a backbone name::

        results = visaebench.evaluate(
            sae=my_sae,
            backbone_name="clip_vitb16",
            activations=shard_paths,
            labels=labels,
            images=val_images,
            mean=mean, std=std,
            device="cuda",
        )

    Args:
        sae: Trained SAE implementing ``encode()`` and ``decode()``.
        backbone: A loaded backbone instance (or ``None``).
        backbone_name: Registry name (e.g. ``"clip_vitb16"``).  Used to
            auto-load the backbone and select the cross-model evaluator
            for monosemanticity.  Ignored if *backbone* is already provided.
        activations: Pre-extracted activations — tensor ``[N, P, D]`` or
            list of shard file paths.
        labels: Integer class labels aligned with *activations*.
        images: PIL images aligned with *activations* (needed for
            monosemanticity).
        mean: Per-dimension normalisation mean, shape ``[D]``.
        std: Scalar normalisation std.
        metrics: ``"all"`` or a list of metric names (see
            :func:`list_metrics`).
        device: Torch device string.
        grid_size: Patch grid ``(H, W)`` for the localization metric.
        seed: Random seed.
        **kwargs: Forwarded to metrics (e.g. ``ood_datasets`` for
            cross-domain).

    Returns:
        :class:`~visaebench.core.results.EvalResults` containing all
        computed :class:`~visaebench.core.types.MetricResult` entries.
    """
    # Resolve backbone from name if needed
    if backbone is None and backbone_name is not None:
        backbone = load_backbone(backbone_name, device=device)

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
        **kwargs,
    )

    print()
    print(results.summary())
    return results


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
    # Metrics
    "METRIC_REGISTRY",
    "MetricRunner",
    "list_metrics",
    # SAE loading
    "GenericSAE",
    "load_sae",
    # Top-level API
    "evaluate",
]
