"""Aggregated evaluation results with display and export helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from visaebench.core.types import DIMENSIONS, MetricResult


class EvalResults:
    """Collection of :class:`MetricResult` objects from a benchmark run.

    Provides helpers for pretty-printing, serialisation, and visualisation.
    """

    def __init__(self, results: list[MetricResult] | None = None) -> None:
        self._results: list[MetricResult] = list(results) if results else []

    # -- container protocol ----------------------------------------------

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    # -- mutation --------------------------------------------------------

    def add(self, result: MetricResult) -> None:
        self._results.append(result)

    # -- access ----------------------------------------------------------

    @property
    def results(self) -> list[MetricResult]:
        return list(self._results)

    def by_dimension(self) -> dict[str, list[MetricResult]]:
        grouped: dict[str, list[MetricResult]] = defaultdict(list)
        for r in self._results:
            grouped[r.dimension].append(r)
        return dict(grouped)

    # -- display ---------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted text table grouped by capability dimension."""
        grouped = self.by_dimension()
        lines: list[str] = []
        sep = "-" * 62

        for dim in DIMENSIONS:
            metrics = grouped.get(dim, [])
            if not metrics:
                continue
            lines.append(sep)
            lines.append(f"  {dim.upper()}")
            lines.append(sep)
            for m in metrics:
                arrow = "\u2191" if m.higher_is_better else "\u2193"
                lines.append(f"    {m.metric_name:<30s} {m.value:>10.4f}  {arrow}")

        lines.append(sep)
        return "\n".join(lines)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise all results to a plain dict (JSON-safe)."""
        out: dict[str, Any] = {}
        for dim in DIMENSIONS:
            out[dim] = {}
        for r in self._results:
            entry: dict[str, Any] = {
                "value": r.value,
                "higher_is_better": r.higher_is_better,
            }
            if r.metadata:
                entry["metadata"] = r.metadata
            out[r.dimension][r.metric_name] = entry
        return out

    def to_dataframe(self):
        """Return a :class:`pandas.DataFrame` with one row per metric.

        Raises :class:`ImportError` if *pandas* is not installed.
        """
        import pandas as pd  # noqa: F811

        rows = [
            {
                "dimension": r.dimension,
                "metric": r.metric_name,
                "value": r.value,
                "higher_is_better": r.higher_is_better,
            }
            for r in self._results
        ]
        return pd.DataFrame(rows)

    # -- visualisation ---------------------------------------------------

    def plot_radar(self, *, title: str = "VISAEBench", ax=None):
        """Draw a radar chart with one axis per capability dimension.

        Each dimension is represented by its *mean* metric value (all metrics
        within that dimension are averaged).  Values are assumed to already be
        on a [0, 1] scale or otherwise comparable.

        Parameters
        ----------
        title:
            Plot title.
        ax:
            Optional *matplotlib* ``Axes`` (must be a polar projection).
            A new figure is created when ``None``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the radar chart.
        """
        import math

        import matplotlib.pyplot as plt
        import numpy as np

        grouped = self.by_dimension()
        labels: list[str] = []
        values: list[float] = []
        for dim in DIMENSIONS:
            metrics = grouped.get(dim, [])
            labels.append(dim.replace("_", " ").title())
            values.append(
                sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
            )

        n = len(labels)
        angles = [i * 2 * math.pi / n for i in range(n)]
        # close the polygon
        values_closed = values + [values[0]]
        angles_closed = angles + [angles[0]]

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        else:
            fig = ax.figure

        ax.plot(angles_closed, values_closed, linewidth=2)
        ax.fill(angles_closed, values_closed, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_title(title, pad=20)

        return fig
