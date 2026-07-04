"""Evaluate a pretrained SAE from the HuggingFace Hub on ImageNet.

What this does
    Downloads a released VISAEBench BatchTopK checkpoint, then runs the
    full 7-metric benchmark on a small ImageNet-val subset and prints the
    aggregated summary table.

Hardware
    Requires a CUDA GPU.  The backbone (CLIP ViT-B/16) plus the cross-model
    evaluator (DINOv2, used by M5 monosemanticity) run on the GPU.

Network / data
    Downloads on first run: the SAE checkpoint from the Hub, the CLIP and
    DINOv2 backbone weights, the ImageNet-1k validation split (streamed from
    the Hub when ``imagenet_path`` is not given), and the OOD datasets
    (EuroSAT, DTD) used by M6.  Activations are cached under
    ``~/.cache/visaebench`` so later runs are much faster.

Runtime
    A few minutes on a modern GPU with ``max_samples=500``.  A full 50k-val
    run takes considerably longer and much more disk for the cache.

Run with
    python examples/01_evaluate_hub_checkpoint.py

This script is import/syntax checked in CI; it is not executed there
because it needs a GPU and network access.
"""

import visaebench
from visaebench.hub import load_sae


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load a pretrained SAE from the Hub.
    #
    # One repo hosts many configs, one per subfolder.  Here we pick the
    # BatchTopK SAE with a 16x dictionary expansion and k=128.
    # ------------------------------------------------------------------
    sae = load_sae(
        "visaebench/clip-vitb16-saes",
        subfolder="batchtopk_16x_k128",
        device="cuda",
    )
    print(sae)

    # ------------------------------------------------------------------
    # 2. Run the benchmark.
    #
    # backbone_name selects the extractor (CLIP ViT-B/16) and, via the
    # cross-model map, the M5 evaluator backbone (DINOv2).  max_samples
    # keeps this quick; drop it to evaluate on the full validation set.
    # imagenet_path=None streams ImageNet-1k from the Hub; pass a local
    # ImageFolder path (val/<synset>/*.JPEG) to use a local copy instead.
    # ------------------------------------------------------------------
    results = visaebench.evaluate(
        sae=sae,
        backbone_name="clip_vitb16",
        imagenet_path=None,
        max_samples=500,
        device="cuda",
    )

    # ``evaluate`` prints the summary too, but we show it explicitly here.
    print(results.summary())


if __name__ == "__main__":
    main()
