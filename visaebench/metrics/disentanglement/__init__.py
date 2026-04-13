"""Disentanglement metrics (M7)."""

from visaebench.metrics.disentanglement.absorption import (
    FeatureAbsorption,
    load_sibling_groups,
)

__all__ = ["FeatureAbsorption", "load_sibling_groups"]
