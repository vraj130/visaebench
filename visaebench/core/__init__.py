"""Core types and interfaces for VISAEBench."""

from visaebench.core.results import EvalResults
from visaebench.core.types import (
    DIMENSIONS,
    BackboneInterface,
    MetricResult,
    SAEInterface,
)

__all__ = [
    "DIMENSIONS",
    "BackboneInterface",
    "EvalResults",
    "MetricResult",
    "SAEInterface",
]
