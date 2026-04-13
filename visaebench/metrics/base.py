"""Abstract base class for all VISAEBench metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from visaebench.core.types import BackboneInterface, MetricResult, SAEInterface


class Metric(ABC):
    """Base class that every benchmark metric must subclass.

    Subclasses declare three class-level attributes and implement
    :meth:`compute`.

    Attributes:
        name: Short identifier used in result dicts (e.g. ``"fvu"``).
        dimension: One of the four capability dimensions defined in
            :data:`visaebench.core.types.DIMENSIONS`.
        higher_is_better: Interpretation hint — ``True`` when larger values
            indicate better SAE quality.
    """

    name: str
    dimension: str
    higher_is_better: bool

    @abstractmethod
    def compute(
        self,
        sae: SAEInterface,
        activations: Any,
        *,
        device: str = "cuda",
        **kwargs,
    ) -> MetricResult:
        """Run the metric and return a single :class:`MetricResult`.

        Args:
            sae: A trained sparse autoencoder exposing ``encode`` / ``decode``.
            activations: Pre-extracted activations.  The concrete type depends
                on the metric — typically a :class:`torch.Tensor` of shape
                ``[N, D]`` or a list of shard file paths.
            device: Torch device string.
            **kwargs: Metric-specific options (labels, batch sizes, etc.).
        """
        ...

    def _make_result(self, value: float, metadata: dict[str, Any] | None = None) -> MetricResult:
        """Helper to construct a :class:`MetricResult` with this metric's attrs."""
        return MetricResult(
            metric_name=self.name,
            dimension=self.dimension,
            value=value,
            higher_is_better=self.higher_is_better,
            metadata=metadata or {},
        )
