"""MAE ViT-B/16 backbone via timm."""

from __future__ import annotations

import timm
import timm.data
import torch
from PIL import Image

from visaebench.backbones.registry import register_backbone

TIMM_MODEL = "vit_base_patch16_224.mae"
PATCH_COUNT = 196  # 14x14 patches
D_MODEL = 768


@register_backbone("mae_vitb16")
class MAEBackbone:
    """MAE ViT-B/16 (encoder only, no masking). Drops CLS token at position 0."""

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self._model = timm.create_model(TIMM_MODEL, pretrained=True)
        self._model.eval().to(device)
        data_cfg = timm.data.resolve_model_data_config(self._model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)

    def _preprocess(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            return images
        if not isinstance(images, (list, tuple)):
            images = [images]
        return torch.stack([
            self._transform(img.convert("RGB") if isinstance(img, Image.Image) else img)
            for img in images
        ])

    @torch.no_grad()
    def extract_activations(
        self,
        images,
        *,
        layer: int = 11,
    ) -> torch.Tensor:
        x = self._preprocess(images).to(self.device)
        # forward_intermediates returns (final, [intermediates])
        # indices are 0-based block indices
        intermediates = self._model.forward_intermediates(
            x, indices=[layer], intermediates_only=True
        )
        hidden = intermediates[0]
        # timm ViTs return [B, 1+num_patches, D] (CLS + patches)
        if hidden.shape[1] == PATCH_COUNT + 1:
            hidden = hidden[:, 1:, :]
        assert hidden.shape[-2:] == (PATCH_COUNT, D_MODEL)
        return hidden
