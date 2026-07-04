"""Tests for the metric registry and MetricRunner selection logic."""

from __future__ import annotations

import pytest

from visaebench.metrics import METRIC_REGISTRY
from visaebench.metrics.base import Metric
from visaebench.metrics.runner import MetricRunner

EXPECTED_ORDER = [
    "localization",
    "fvu",
    "downstream_preservation",
    "sparse_probing",
    "monosemanticity",
    "cross_domain",
    "absorption",
]


def test_registry_keys_exact_order():
    assert list(METRIC_REGISTRY.keys()) == EXPECTED_ORDER


def test_all_registry_values_are_metric_subclasses():
    for key, cls in METRIC_REGISTRY.items():
        assert issubclass(cls, Metric), key


def test_runner_rejects_unknown_metric():
    with pytest.raises(KeyError):
        MetricRunner(metrics=["nonexistent_metric"])


def test_runner_all_selects_every_metric():
    runner = MetricRunner(metrics="all")
    assert runner.metric_names == EXPECTED_ORDER


def test_runner_explicit_subset():
    runner = MetricRunner(metrics=["fvu", "localization"])
    assert runner.metric_names == ["fvu", "localization"]
