"""DINOv2 ViT-B/14 backbone via torch.hub."""

from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image

from visaebench.backbones.registry import register_backbone

PATCH_COUNT = 256  # 16x16 patches from 224x224 with patch_size=14
D_MODEL = 768

# DINOv2 ImageNet normalization
_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _preprocess(images) -> torch.Tensor:
    """Convert a list of PIL images or a tensor batch to a normalised tensor."""
    if isinstance(images, torch.Tensor):
        return images
    if not isinstance(images, (list, tuple)):
        images = [images]
    return torch.stack([_TRANSFORM(img.convert("RGB") if isinstance(img, Image.Image) else img) for img in images])


@register_backbone("dinov2_vitb14")
class DINOv2Backbone:
    """DINOv2 ViT-B/14.  Drops CLS token at position 0.

    ``facebook/dinov2-base`` has no register tokens, so the sequence is
    ``[CLS, patch_0, ..., patch_255]`` (257 tokens total).
    """

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self._model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        )
        self._model.eval().to(device)

    @torch.no_grad()
    def extract_activations(
        self,
        images,
        *,
        layer: int = 11,
    ) -> torch.Tensor:
        x = _preprocess(images).to(self.device)
        # get_intermediate_layers with n=[idx] returns specific layer outputs
        # Each output is [B, 1+num_patches, D] (includes CLS)
        out = self._model.get_intermediate_layers(x, n=[layer], reshape=False)
        hidden = out[0]
        # For dinov2-base (no registers) this is already patch-only
        # get_intermediate_layers strips CLS by default
        if hidden.shape[1] == PATCH_COUNT + 1:
            hidden = hidden[:, 1:, :]
        assert hidden.shape[-2:] == (PATCH_COUNT, D_MODEL)
        return hidden
