"""CLIP ViT-B/16 backbone via HuggingFace transformers."""

from __future__ import annotations

import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from visaebench.backbones.registry import register_backbone

MODEL_ID = "openai/clip-vit-base-patch16"
PATCH_COUNT = 196  # 14x14 patches from 224x224 with patch_size=16
D_MODEL = 768


@register_backbone("clip_vitb16")
class CLIPBackbone:
    """CLIP ViT-B/16. Drops CLS token at position 0."""

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self._processor = CLIPImageProcessor.from_pretrained(MODEL_ID)
        self._model = CLIPVisionModel.from_pretrained(MODEL_ID)
        self._model.eval().to(device)

    @torch.no_grad()
    def extract_activations(
        self,
        images,
        *,
        layer: int = 11,
    ) -> torch.Tensor:
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs, output_hidden_states=True)
        # hidden_states[0] is embedding output; layer N is at index N+1
        hidden = outputs.hidden_states[layer + 1]
        # Drop CLS at position 0
        patch_tokens = hidden[:, 1:, :]
        assert patch_tokens.shape[-2:] == (PATCH_COUNT, D_MODEL)
        return patch_tokens
