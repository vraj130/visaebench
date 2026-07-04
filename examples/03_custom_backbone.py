"""Evaluate an SAE against a ViT backbone that is not in the registry.

What this does
    Shows how to wrap an arbitrary HuggingFace or timm Vision Transformer
    with ``CustomBackbone`` and hand that instance to ``visaebench.evaluate``
    for activation extraction, instead of relying on a registered backbone
    name.

Hardware
    Requires a CUDA GPU for a real run (the backbone and the M5 cross-model
    evaluator run on the GPU).

Network / data
    Downloads on first run: the chosen ViT weights (HuggingFace or timm),
    ImageNet-val, and the OOD datasets used by M6.

Runtime
    A few minutes on a GPU with ``max_samples=500``.

Run with
    python examples/03_custom_backbone.py

This script is import/syntax checked in CI; it is not executed there
because it needs a GPU and network access.
"""

import visaebench
from visaebench.backbones import CustomBackbone


def build_hf_backbone() -> CustomBackbone:
    """Wrap a HuggingFace ViT.

    ``num_prefix_tokens`` is the number of leading non-spatial tokens to
    drop from the hidden states (CLS, distillation, register tokens, ...).
    A plain ViT-B/16 has a single CLS token, so the default of 1 is
    correct here.  For a model with more prefix tokens (for example DeiT
    with a distillation token, or DINOv2 with register tokens) pass the
    matching count so only true patch tokens remain.
    """
    return CustomBackbone(
        "google/vit-base-patch16-224",
        source="huggingface",
        num_prefix_tokens=1,
        device="cuda",
    )


def build_timm_backbone() -> CustomBackbone:
    """Wrap a timm ViT.

    timm handles its own preprocessing transform, resolved from the model
    config.  ``num_prefix_tokens`` has the same meaning as above.
    """
    return CustomBackbone(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        source="timm",
        num_prefix_tokens=1,
        device="cuda",
    )


def main() -> None:
    sae = ...  # your SAE: any object with encode(x) and decode(z)

    backbone = build_hf_backbone()
    # A timm alternative:
    # backbone = build_timm_backbone()

    # ------------------------------------------------------------------
    # Pass the instance via ``backbone=``.
    #
    # ``backbone_name`` is still required even when you pass a custom
    # instance: it is not used for extraction (the instance is), but it
    # is the activation cache key AND it selects the M5 cross-model
    # evaluator via the internal cross-model map.  Pick the registered
    # name whose cross-model pairing you want; here we reuse
    # "clip_vitb16", which maps to DINOv2 as the M5 evaluator.
    # ------------------------------------------------------------------
    results = visaebench.evaluate(
        sae=sae,
        backbone=backbone,
        backbone_name="clip_vitb16",
        imagenet_path=None,
        max_samples=500,
        device="cuda",
    )
    print(results.summary())


if __name__ == "__main__":
    main()
