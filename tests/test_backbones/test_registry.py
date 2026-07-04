"""Tests for the backbone registry (no real backbone instantiation)."""

from __future__ import annotations

import pytest

from visaebench.backbones.registry import list_backbones, load_backbone

EXPECTED_BACKBONES = {
    "clip_vitb16",
    "dinov2_vitb14",
    "siglip_vitb16",
    "mae_vitb16",
    "deit_vitb16",
}


def test_list_backbones_exact_set():
    # importing the package triggers side-effect registration of all backbones
    import visaebench.backbones  # noqa: F401

    assert set(list_backbones()) == EXPECTED_BACKBONES


def test_load_unknown_backbone_raises_keyerror():
    with pytest.raises(KeyError) as exc:
        load_backbone("nonexistent", device="cpu")
    # message should name the offending key and list available ones
    assert "nonexistent" in str(exc.value)
