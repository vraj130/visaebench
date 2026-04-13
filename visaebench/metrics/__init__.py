"""VISAEBench evaluation metrics."""

from visaebench.metrics.base import Metric
from visaebench.metrics.concept_detection.cross_domain import CrossDomainGeneralization
from visaebench.metrics.concept_detection.monosemanticity import MonosemanticityScore
from visaebench.metrics.concept_detection.sparse_probing import SparseProbing
from visaebench.metrics.disentanglement.absorption import FeatureAbsorption
from visaebench.metrics.reconstruction.downstream import DownstreamPreservation
from visaebench.metrics.reconstruction.fvu import FVU
from visaebench.metrics.spatial_coherence.localization import FeatureLocalization

METRIC_REGISTRY: dict[str, type[Metric]] = {
    "fvu": FVU,
    "downstream_preservation": DownstreamPreservation,
    "sparse_probing": SparseProbing,
    "monosemanticity": MonosemanticityScore,
    "cross_domain": CrossDomainGeneralization,
    "localization": FeatureLocalization,
    "absorption": FeatureAbsorption,
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
