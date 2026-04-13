"""Vision Transformer backbone adapters for activation extraction."""

# Import backbone modules to trigger registration via @register_backbone
from visaebench.backbones import clip  # noqa: F401
from visaebench.backbones import deit  # noqa: F401
from visaebench.backbones import dinov2  # noqa: F401
from visaebench.backbones import mae  # noqa: F401
from visaebench.backbones import siglip  # noqa: F401
from visaebench.backbones.custom import CustomBackbone
from visaebench.backbones.registry import list_backbones, load_backbone

__all__ = [
    "CustomBackbone",
    "list_backbones",
    "load_backbone",
]
