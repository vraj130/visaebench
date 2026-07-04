"""Tests for EvalResults container, grouping, summary and JSON export."""

from __future__ import annotations

import json

from visaebench.core.results import EvalResults
from visaebench.core.types import MetricResult


def _sample_results():
    return [
        MetricResult("fvu", "reconstruction", 0.05, higher_is_better=False),
        MetricResult("downstream_preservation", "reconstruction", 0.9, higher_is_better=True),
        MetricResult("sparse_probing_auc", "concept_detection", 0.7, higher_is_better=True,
                     metadata={"num_images": 100}),
    ]


def test_len_iter_and_add():
    er = EvalResults()
    assert len(er) == 0
    for r in _sample_results():
        er.add(r)
    assert len(er) == 3
    names = [r.metric_name for r in er]  # iteration
    assert names == ["fvu", "downstream_preservation", "sparse_probing_auc"]


def test_by_dimension_grouping():
    er = EvalResults(_sample_results())
    grouped = er.by_dimension()
    assert set(grouped.keys()) == {"reconstruction", "concept_detection"}
    assert len(grouped["reconstruction"]) == 2
    assert len(grouped["concept_detection"]) == 1


def test_summary_contains_uppercased_dimensions():
    er = EvalResults(_sample_results())
    text = er.summary()
    assert isinstance(text, str)
    assert len(text) > 0
    assert "RECONSTRUCTION" in text
    assert "CONCEPT_DETECTION" in text
    # dimensions with no results should not appear
    assert "SPATIAL_COHERENCE" not in text
    assert "fvu" in text


def test_to_dict_json_roundtrip(tmp_path):
    er = EvalResults(_sample_results())
    d = er.to_dict()

    # metrics land under their dimension buckets
    assert d["reconstruction"]["fvu"]["value"] == 0.05
    assert d["concept_detection"]["sparse_probing_auc"]["metadata"]["num_images"] == 100

    path = tmp_path / "results.json"
    with open(path, "w") as f:
        json.dump(d, f)
    with open(path) as f:
        loaded = json.load(f)

    assert loaded == d
