"""VISAEBench evaluation metrics."""

from visaebench.metrics.base import Metric
from visaebench.metrics.concept_detection.cross_domain import CrossDomainGeneralization
from visaebench.metrics.concept_detection.monosemanticity import MonosemanticityScore
from visaebench.metrics.concept_detection.sparse_probing import SparseProbing
from visaebench.metrics.disentanglement.absorption import FeatureAbsorption
from visaebench.metrics.reconstruction.downstream import DownstreamPreservation
from visaebench.metrics.reconstruction.fvu import FVU
from visaebench.metrics.spatial_coherence.localization import FeatureLocalization

# Canonical M1–M7 ordering. The runner iterates this dict in order, so this
# is also the default execution / radar-chart sequence.
METRIC_REGISTRY: dict[str, type[Metric]] = {
    "localization": FeatureLocalization,                # M1: spatial_coherence
    "fvu": FVU,                                         # M2: reconstruction
    "downstream_preservation": DownstreamPreservation,  # M3: reconstruction
    "sparse_probing": SparseProbing,                    # M4: concept_detection
    "monosemanticity": MonosemanticityScore,            # M5: concept_detection
    "cross_domain": CrossDomainGeneralization,          # M6: concept_detection
    "absorption": FeatureAbsorption,                    # M7: disentanglement
}

__all__ = [
    "METRIC_REGISTRY",
    "Metric",
    "CrossDomainGeneralization",
    "DownstreamPreservation",
    "FeatureAbsorption",
    "FeatureLocalization",
    "FVU",
    "MonosemanticityScore",
    "SparseProbing",
]
