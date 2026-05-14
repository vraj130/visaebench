"""Core type definitions and interfaces for VISAEBench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# Capability dimensions
# ---------------------------------------------------------------------------

DIMENSIONS = (
    "reconstruction",
    "concept_detection",
    "spatial_coherence",
    "disentanglement",
)

# ---------------------------------------------------------------------------
# SAE interface
# ---------------------------------------------------------------------------


@runtime_checkable
class SAEInterface(Protocol):
    """Structural interface for a Sparse Autoencoder.

    Any object that implements ``encode`` and ``decode`` with the correct
    signatures satisfies this protocol — no inheritance required.
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode patch activations into sparse latent codes.

        Args:
            x: Activations of shape ``[batch, d_model]``.

        Returns:
            Sparse latent codes of shape ``[batch, d_sae]``.
        """
        ...

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent codes back to activation space.

        Args:
            z: Sparse codes of shape ``[batch, d_sae]``.

        Returns:
            Reconstructed activations of shape ``[batch, d_model]``.
        """
        ...


# ---------------------------------------------------------------------------
# Backbone interface
# ---------------------------------------------------------------------------


@runtime_checkable
class BackboneInterface(Protocol):
    """Structural interface for a Vision Transformer backbone.

    Implementations extract intermediate patch-level activations from a ViT.
    """

    def extract_activations(
        self,
        images: torch.Tensor,
        *,
        layer: int = -1,
    ) -> torch.Tensor:
        """Extract patch-level activations from a transformer layer.

        Args:
            images: Batch of images, shape ``[B, C, H, W]``.
            layer: 0-indexed transformer block. ``-1`` means the last block.

        Returns:
            Patch activations of shape ``[B, num_patches, d_model]``.
        """
        ...


# ---------------------------------------------------------------------------
# Metric result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """Container for a single metric measurement.

    Attributes:
        metric_name: Short identifier, e.g. ``"fvu"`` or ``"moran_i"``.
        dimension: One of :data:`DIMENSIONS`.
        value: The scalar metric value.
        higher_is_better: Interpretation hint for downstream consumers.
        metadata: Arbitrary extra information (timings, per-class scores, …).
    """

    metric_name: str
    dimension: str
    value: float
    higher_is_better: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSIONS:
            raise ValueError(
                f"dimension must be one of {DIMENSIONS}, got {self.dimension!r}"
            )

    @property
    def name(self) -> str:
        """Alias for :attr:`metric_name`."""
        return self.metric_name
