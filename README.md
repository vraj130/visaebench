# VISAEBench

VISAEBench is a benchmark for evaluating Sparse Autoencoders (SAEs) trained on Vision Transformer (ViT) patch activations. It scores an SAE across four interpretability dimensions (reconstruction, concept detection, spatial coherence, disentanglement) using 7 metrics (M1 through M7). It supports cross-backbone evaluation across CLIP, DINOv2, SigLIP, MAE, and DeiT (all ViT-B/16 or ViT-B/14).

## Installation

```bash
# From PyPI (after v0.1.0 release)
pip install visaebench

# From source (development)
git clone https://github.com/visaebench/visaebench && cd visaebench
pip install -e .
```

## Quickstart

```python
import visaebench
from visaebench.hub import load_sae

sae = load_sae("visaebench/clip-vitb16-saes", subfolder="batchtopk_16x_k128")
results = visaebench.evaluate(
    sae=sae,
    backbone_name="clip_vitb16",
    imagenet_path="/path/to/imagenet/val",  # or None to stream from HF
)
for r in results:
    print(f"{r.name:30s} {r.value:.4f}")
```

## Available SAE checkpoints

One HuggingFace repo per backbone; each holds 12 configs as subfolders (`batchtopk_{8x,16x,32x}_k{64,128,192,256}`).

| Backbone | HuggingFace repo |
|---|---|
| CLIP ViT-B/16 | `visaebench/clip-vitb16-saes` |
| DINOv2 ViT-B/14 | `visaebench/dinov2-vitb14-saes` |
| SigLIP ViT-B/16 | `visaebench/siglip-vitb16-saes` |
| MAE ViT-B/16 | `visaebench/mae-vitb16-saes` |
| DeiT ViT-B/16 | `visaebench/deit-vitb16-saes` |

## Metrics

| # | Registry key | Dimension | Description |
|---|---|---|---|
| M1 | `localization` | spatial_coherence | Per-feature spatial coherence on patch grid |
| M2 | `fvu` | reconstruction | Fraction of variance unexplained |
| M3 | `downstream_preservation` | reconstruction | Linear probe accuracy after reconstruction vs raw |
| M4 | `sparse_probing` | concept_detection | Sparse logistic probe AUC over class labels |
| M5 | `monosemanticity` | concept_detection | Cross-model feature consistency |
| M6 | `cross_domain` | concept_detection | Generalization to OOD datasets (EuroSAT, DTD) |
| M7 | `absorption` | disentanglement | WordNet-hierarchy concept absorption rate |

## Citation

```bibtex
@inproceedings{visaebench2026,
  title  = {VISAEBench: A Benchmark for Vision Sparse Autoencoders},
  author = {},
  booktitle = {},
  year   = {2026},
  note   = {TODO: }
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
