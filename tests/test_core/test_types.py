"""Tests for core type definitions (MetricResult, SAEInterface)."""

from __future__ import annotations

import pytest
import torch

from visaebench.core.types import DIMENSIONS, MetricResult, SAEInterface


def test_metric_result_rejects_bad_dimension():
    with pytest.raises(ValueError):
        MetricResult(
            metric_name="x",
            dimension="not_a_dim",
            value=0.0,
            higher_is_better=True,
        )


def test_metric_result_accepts_all_valid_dimensions():
    for dim in DIMENSIONS:
        r = MetricResult(
            metric_name="m",
            dimension=dim,
            value=1.0,
            higher_is_better=False,
        )
        assert r.dimension == dim


def test_metric_result_name_alias():
    r = MetricResult(
        metric_name="fvu",
        dimension="reconstruction",
        value=0.1,
        higher_is_better=False,
    )
    assert r.name == r.metric_name == "fvu"


class _DuckSAE:
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z


class _EncoderOnly:
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_duck_typed_sae_passes_protocol():
    assert isinstance(_DuckSAE(), SAEInterface)


def test_object_missing_decode_fails_protocol():
    assert not isinstance(_EncoderOnly(), SAEInterface)
