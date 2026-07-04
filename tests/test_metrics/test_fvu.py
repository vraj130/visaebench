"""CPU-only synthetic tests for metrics that run without ImageNet.

Covers FVU (M2) exactly, plus DownstreamPreservation (M3) and
SparseProbing (M4) driven on tiny synthetic activations and labels.
Metrics that genuinely need real images / a backbone are not exercised here.
"""

from __future__ import annotations

import numpy as np
import torch

from visaebench.core.types import SAEInterface
from visaebench.metrics.concept_detection.sparse_probing import SparseProbing
from visaebench.metrics.reconstruction.downstream import DownstreamPreservation
from visaebench.metrics.reconstruction.fvu import FVU


class _IdentitySAE:
    """decode(encode(x)) == x, so the residual is exactly zero."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z


class _MeanSAE:
    """decode returns the per-column mean broadcast to the input shape.

    With approximately zero per-column means the residual variance equals
    the input variance, driving FVU toward 1.0.
    """

    def __init__(self, mu: torch.Tensor) -> None:
        self._mu = mu

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._mu.expand_as(z)


# ---------------------------------------------------------------------------
# FVU
# ---------------------------------------------------------------------------


def test_fvu_class_attributes():
    assert FVU.higher_is_better is False
    assert FVU.name == "fvu"
    assert FVU.dimension == "reconstruction"


def test_fvu_perfect_reconstruction_is_zero():
    torch.manual_seed(0)
    acts = torch.randn(256, 16)
    sae = _IdentitySAE()
    assert isinstance(sae, SAEInterface)

    result = FVU().compute(sae, acts, device="cpu", mean=None, std=None)
    assert abs(result.value) < 1e-4
    assert result.dimension == "reconstruction"
    assert abs(result.metadata["explained_variance"] - 1.0) < 1e-4


def test_fvu_mean_predictor_is_about_one():
    torch.manual_seed(0)
    acts = torch.randn(1024, 16)  # per-column means ~ N(0, 1/1024)
    mu = acts.mean(dim=0)
    sae = _MeanSAE(mu)
    assert isinstance(sae, SAEInterface)

    result = FVU().compute(sae, acts, device="cpu", mean=None, std=None)
    assert abs(result.value - 1.0) < 0.02


# ---------------------------------------------------------------------------
# DownstreamPreservation (M3): identity SAE preserves everything -> ratio 1.0
# ---------------------------------------------------------------------------


def test_downstream_identity_preservation_is_one():
    torch.manual_seed(0)
    n_per_class, n_classes, P, D = 10, 4, 4, 16
    acts = torch.randn(n_per_class * n_classes, P, D)
    labels = np.repeat(np.arange(n_classes), n_per_class).astype(np.int64)

    result = DownstreamPreservation().compute(
        _IdentitySAE(), acts, device="cpu", labels=labels, seed=42,
    )
    # identical original and reconstructed features -> equal accuracy
    assert abs(result.value - 1.0) < 1e-6
    assert result.dimension == "reconstruction"
    assert result.higher_is_better is True
    assert result.metadata["num_images"] == n_per_class * n_classes


# ---------------------------------------------------------------------------
# SparseProbing (M4): runs on synthetic data, value is a bounded AUC
# ---------------------------------------------------------------------------


def test_sparse_probing_runs_on_synthetic():
    torch.manual_seed(0)
    n_per_class, n_classes, P, D = 12, 3, 4, 16
    acts = torch.randn(n_per_class * n_classes, P, D)
    labels = np.repeat(np.arange(n_classes), n_per_class).astype(np.int64)

    metric = SparseProbing(k_values=[2, 4, 8])
    result = metric.compute(_IdentitySAE(), acts, device="cpu", labels=labels, seed=42)

    assert result.name == "sparse_probing_auc"
    assert result.dimension == "concept_detection"
    assert 0.0 <= result.value <= 1.0
    assert result.value == result.value  # not NaN
    assert result.metadata["num_images"] == n_per_class * n_classes
