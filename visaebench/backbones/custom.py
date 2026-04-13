"""Generic adapter that wraps any HuggingFace or timm ViT model."""

from __future__ import annotations

from typing import Literal

import torch
from PIL import Image


class CustomBackbone:
    """Wrap an arbitrary HuggingFace or timm ViT as a VISAEBench backbone.

    Parameters
    ----------
    model_name:
        A HuggingFace model ID (e.g. ``"google/vit-base-patch16-224"``)
        or a timm model name (e.g. ``"vit_small_patch16_224"``).
    source:
        ``"huggingface"`` or ``"timm"``.
    num_prefix_tokens:
        Number of non-spatial prefix tokens to drop (CLS, distillation, etc.).
        Defaults to 1 (CLS only).
    device:
        Torch device string.
    """

    def __init__(
        self,
        model_name: str,
        *,
        source: Literal["huggingface", "timm"] = "huggingface",
        num_prefix_tokens: int = 1,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self._source = source
        self._num_prefix = num_prefix_tokens

        if source == "huggingface":
            self._init_huggingface(model_name)
        elif source == "timm":
            self._init_timm(model_name)
        else:
            raise ValueError(f"source must be 'huggingface' or 'timm', got {source!r}")

    # -- init helpers ----------------------------------------------------

    def _init_huggingface(self, model_name: str) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval().to(self.device)

    def _init_timm(self, model_name: str) -> None:
        import timm
        import timm.data

        self._model = timm.create_model(model_name, pretrained=True)
        self._model.eval().to(self.device)
        data_cfg = timm.data.resolve_model_data_config(self._model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)

    # -- preprocessing ---------------------------------------------------

    def _preprocess_timm(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            return images
        if not isinstance(images, (list, tuple)):
            images = [images]
        return torch.stack([
            self._transform(img.convert("RGB") if isinstance(img, Image.Image) else img)
            for img in images
        ])

    # -- extract ---------------------------------------------------------

    @torch.no_grad()
    def extract_activations(
        self,
        images,
        *,
        layer: int = -1,
    ) -> torch.Tensor:
        if self._source == "huggingface":
            return self._extract_hf(images, layer)
        return self._extract_timm(images, layer)

    def _extract_hf(self, images, layer: int) -> torch.Tensor:
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1] if layer >= 0 else outputs.hidden_states[layer]
        if self._num_prefix > 0:
            hidden = hidden[:, self._num_prefix:, :]
        return hidden

    def _extract_timm(self, images, layer: int) -> torch.Tensor:
        x = self._preprocess_timm(images).to(self.device)
        if layer < 0:
            # Use the model's forward and get last hidden state
            features = self._model.forward_features(x)
            if self._num_prefix > 0:
                features = features[:, self._num_prefix:, :]
            return features
        intermediates = self._model.forward_intermediates(
            x, indices=[layer], intermediates_only=True
        )
        hidden = intermediates[0]
        if self._num_prefix > 0:
            hidden = hidden[:, self._num_prefix:, :]
        return hidden
