"""MetricRunner: orchestrates multiple metrics with shared computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm

from visaebench.core.results import EvalResults
from visaebench.core.types import BackboneInterface, MetricResult, SAEInterface
from visaebench.metrics import METRIC_REGISTRY
from visaebench.metrics.base import Metric

# Cross-model map for monosemanticity: SAE backbone → evaluator backbone.
CROSS_MODEL_MAP: dict[str, str] = {
    "clip_vitb16": "dinov2_vitb14",
    "dinov2_vitb14": "clip_vitb16",
    "siglip_vitb16": "dinov2_vitb14",
    "mae_vitb16": "clip_vitb16",
    "deit_vitb16": "clip_vitb16",
}


class MetricRunner:
    """Run multiple VISAEBench metrics with shared activations.

    Parameters
    ----------
    metrics:
        Metric names to run.  ``"all"`` runs every registered metric.
    device:
        Torch device string.
    """

    def __init__(
        self,
        metrics: str | Sequence[str] = "all",
        device: str = "cuda",
    ) -> None:
        if metrics == "all":
            self.metric_names = list(METRIC_REGISTRY.keys())
        else:
            for m in metrics:
                if m not in METRIC_REGISTRY:
                    raise KeyError(
                        f"Unknown metric {m!r}. "
                        f"Available: {sorted(METRIC_REGISTRY)}"
                    )
            self.metric_names = list(metrics)
        self.device = device

    def run(
        self,
        sae: SAEInterface,
        *,
        backbone: BackboneInterface | None = None,
        backbone_name: str | None = None,
        activations: torch.Tensor | list[str | Path] | None = None,
        labels: np.ndarray | None = None,
        images: Sequence | None = None,
        mean: torch.Tensor | None = None,
        std: float | None = None,
        grid_size: tuple[int, int] = (14, 14),
        seed: int = 42,
        **kwargs,
    ) -> EvalResults:
        """Run all selected metrics and return aggregated results.

        Args:
            sae: Trained SAE.
            backbone: Loaded backbone instance (for monosemanticity / cross-domain).
            backbone_name: Registry name of the backbone (used to auto-select
                cross-model evaluator).
            activations: Pre-extracted ImageNet activations — tensor ``[N, P, D]``
                or list of shard file paths.
            labels: Integer class labels aligned with *activations*.
            images: PIL images aligned with *activations* (needed for
                monosemanticity).
            mean: Per-dimension normalisation mean.
            std: Scalar normalisation std.
            grid_size: Patch grid dimensions for the localization metric.
            seed: Random seed.
            **kwargs: Forwarded to individual metrics.
        """
        results = EvalResults()

        for name in tqdm(self.metric_names, desc="VISAEBench"):
            metric_cls = METRIC_REGISTRY[name]
            # Pass grid_size to localization metric constructor
            if name == "localization":
                metric = metric_cls(grid_size=grid_size)
            else:
                metric = metric_cls()

            try:
                result = self._run_one(
                    metric=metric,
                    sae=sae,
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
                results.add(result)
            except Exception as e:
                tqdm.write(f"  [{name}] FAILED: {e}")

            # Free GPU memory between metrics
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------

    def _run_one(
        self,
        metric: Metric,
        *,
        sae: SAEInterface,
        backbone: BackboneInterface | None,
        backbone_name: str | None,
        activations: Any,
        labels: np.ndarray | None,
        images: Sequence | None,
        mean: torch.Tensor | None,
        std: float | None,
        grid_size: tuple[int, int],
        seed: int,
        **kwargs,
    ) -> MetricResult:
        """Dispatch a single metric with the correct kwargs."""
        name = metric.name
        common: dict[str, Any] = {
            "device": self.device,
            "mean": mean,
            "std": std,
            "seed": seed,
        }

        if name == "fvu":
            return metric.compute(sae, activations, **common, **kwargs)

        if name == "downstream_preservation":
            return metric.compute(sae, activations, labels=labels, **common, **kwargs)

        if name == "sparse_probing_auc":
            return metric.compute(sae, activations, labels=labels, **common, **kwargs)

        if name == "monosemanticity":
            # Auto-select cross-model evaluator
            evaluator = self._get_cross_evaluator(backbone_name)
            return metric.compute(
                sae, activations,
                images=images,
                evaluator_backbone=evaluator,
                **common, **kwargs,
            )

        if name == "cross_domain":
            ood = kwargs.pop("ood_datasets", None)
            if ood is None:
                tqdm.write("  [cross_domain] Skipped: no ood_datasets provided")
                raise ValueError("ood_datasets required for cross_domain metric")
            return metric.compute(
                sae, activations,
                ood_datasets=ood,
                backbone=backbone,
                **common, **kwargs,
            )

        if name == "feature_localization":
            return metric.compute(
                sae, activations,
                **common, **kwargs,
            )

        if name == "absorption_rate":
            return metric.compute(sae, activations, labels=labels, **common, **kwargs)

        # Fallback: pass everything we have
        return metric.compute(sae, activations, **common, **kwargs)

    def _get_cross_evaluator(self, backbone_name: str | None) -> BackboneInterface:
        """Load the cross-model evaluator backbone for monosemanticity."""
        from visaebench.backbones import load_backbone

        if backbone_name and backbone_name in CROSS_MODEL_MAP:
            eval_name = CROSS_MODEL_MAP[backbone_name]
        else:
            # Default to DINOv2 as evaluator
            eval_name = "dinov2_vitb14"

        return load_backbone(eval_name, device=self.device)


def list_metrics() -> dict[str, list[str]]:
    """Return metric names grouped by capability dimension."""
    from visaebench.core.types import DIMENSIONS

    grouped: dict[str, list[str]] = {d: [] for d in DIMENSIONS}
    for name, cls in METRIC_REGISTRY.items():
        grouped[cls.dimension].append(name)
    return grouped
