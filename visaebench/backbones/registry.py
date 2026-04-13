"""Backbone registry: maps string names to loader functions."""

from __future__ import annotations

from typing import Callable

from visaebench.core.types import BackboneInterface

# name -> callable(device) -> BackboneInterface
_REGISTRY: dict[str, Callable[..., BackboneInterface]] = {}


def register_backbone(name: str):
    """Decorator that registers a backbone loader under *name*.

    Usage::

        @register_backbone("my_vit")
        class MyViT:
            def __init__(self, *, device="cuda"): ...
            def extract_activations(self, images, *, layer=-1): ...
    """

    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def load_backbone(name: str, *, device: str = "cuda") -> BackboneInterface:
    """Instantiate a registered backbone by name.

    Args:
        name: Registry key (e.g. ``"clip_vitb16"``).
        device: Torch device string.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown backbone {name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](device=device)


def list_backbones() -> list[str]:
    """Return sorted list of registered backbone names."""
    return sorted(_REGISTRY)
