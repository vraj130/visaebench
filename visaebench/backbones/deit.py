"""DeiT-B/16 backbone via timm."""

from __future__ import annotations

import timm
import timm.data
import torch
from PIL import Image

from visaebench.backbones.registry import register_backbone

TIMM_MODEL = "deit_base_patch16_224"
PATCH_COUNT = 196  # 14x14 patches
D_MODEL = 768


@register_backbone("deit_vitb16")
class DeiTBackbone:
    """DeiT ViT-B/16. Drops CLS and distillation tokens (positions 0 and 1)."""

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
        intermediates = self._model.forward_intermediates(
            x, indices=[layer], intermediates_only=True
        )
        hidden = intermediates[0]
        # DeiT has CLS + distillation token + 196 patches = 198 tokens
        num_prefix = hidden.shape[1] - PATCH_COUNT
        if num_prefix > 0:
            hidden = hidden[:, num_prefix:, :]
        assert hidden.shape[-2:] == (PATCH_COUNT, D_MODEL)
        return hidden
