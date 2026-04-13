"""SigLIP ViT-B/16 backbone via HuggingFace transformers."""

from __future__ import annotations

import torch
from transformers import AutoProcessor, SiglipVisionModel

from visaebench.backbones.registry import register_backbone

MODEL_ID = "google/siglip-base-patch16-224"
PATCH_COUNT = 196  # 14x14 patches
D_MODEL = 768


@register_backbone("siglip_vitb16")
class SigLIPBackbone:
    """SigLIP ViT-B/16. Has NO CLS token — all tokens are patch tokens."""

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = SiglipVisionModel.from_pretrained(MODEL_ID)
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
        # SigLIP has no CLS token — all 196 tokens are spatial
        patch_tokens = outputs.hidden_states[layer + 1]
        assert patch_tokens.shape[-2:] == (PATCH_COUNT, D_MODEL)
        return patch_tokens
