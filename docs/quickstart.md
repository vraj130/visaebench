# Quickstart

VISAEBench scores a Sparse Autoencoder (SAE) trained on Vision Transformer
(ViT) patch activations across four capability dimensions
(reconstruction, concept detection, spatial coherence, disentanglement)
using 7 metrics (M1 to M7). This page covers installation, the one-call
`evaluate` flow, the released checkpoints, and how caching works.

## Installation

Editable install from the repo root:

```bash
pip install -e .
```

Tests use pytest, which is not pinned in `pyproject.toml`; install it
separately if you want to run the suite:

```bash
pip install pytest
pytest
```

## The one-call flow

The entry point is `visaebench.evaluate`. The minimal usage is a few
lines: load a backbone by name, point at ImageNet, pass your SAE.

```python
import visaebench
from visaebench.hub import load_sae

sae = load_sae("visaebench/clip-vitb16-saes", subfolder="batchtopk_16x_k128")

results = visaebench.evaluate(
    sae=sae,
    backbone_name="clip_vitb16",
    imagenet_path="/data/imagenet/val",  # or None to stream imagenet-1k from HF
    device="cuda",
)
print(results.summary())
```

With nothing else specified, `evaluate` does the following:

1. Loads the named backbone (`clip_vitb16`, `dinov2_vitb14`,
   `siglip_vitb16`, `mae_vitb16`, or `deit_vitb16`).
2. Loads ImageNet validation lazily, from `imagenet_path` (a torchvision
   `ImageFolder` layout, `val/<synset>/*.JPEG`) or, when `imagenet_path`
   is `None`, by streaming the HuggingFace `imagenet-1k` validation split.
3. Extracts patch activations and caches them as fp16 shards (see
   [Caching](#caching) below).
4. Computes a per-dimension mean and a scalar std for SAE input
   normalisation, in a single streaming pass, cached next to the shards.
5. Auto-loads the cross-model evaluator for M5 monosemanticity (for
   example DINOv2 when the SAE backbone is CLIP).
6. Auto-downloads the OOD datasets for M6. The defaults are EuroSAT and
   DTD.
7. Auto-infers the patch `grid_size` from the backbone's patch count.
8. Runs all 7 metrics and returns an aggregated `EvalResults`. A failure
   in one metric is caught and logged; it does not abort the whole run.

### Useful keyword arguments

- `max_samples`: subsample ImageNet val (default: full 50k). Handy for
  quick iteration.
- `layer`: transformer block index to extract (default 11, the
  second-to-last block on a 12-layer ViT-Base).
- `cache_dir`: override the activation cache root.
- `ood_datasets`: a list of names (for example `["eurosat"]`), an empty
  list `[]` to skip OOD entirely, or a full dict matching the
  `CrossDomainGeneralization.compute` schema.
- `ood_max_images`: cap per OOD dataset (default 10000).
- `metrics`: `"all"` or a list of registry keys, for example
  `["fvu", "sparse_probing"]`.
- `activations`, `labels`, `images`, `mean`, `std`: bypass the
  auto-extraction path by passing pre-computed values.
- `grid_size`: override the auto-inferred patch grid `(H, W)`.

## Released checkpoints

Checkpoints live on the HuggingFace Hub, one repo per backbone. Each repo
holds 12 configs as subfolders, covering three dictionary expansion
factors and four sparsity (k) levels: `batchtopk_{8x,16x,32x}_k{64,128,192,256}`.

| Backbone | HF repo | Subfolders |
|---|---|---|
| CLIP ViT-B/16 | `visaebench/clip-vitb16-saes` | `batchtopk_{8x,16x,32x}_k{64,128,192,256}` |
| DINOv2 ViT-B/14 | `visaebench/dinov2-vitb14-saes` | `batchtopk_{8x,16x,32x}_k{64,128,192,256}` |
| SigLIP ViT-B/16 | `visaebench/siglip-vitb16-saes` | `batchtopk_{8x,16x,32x}_k{64,128,192,256}` |
| MAE ViT-B/16 | `visaebench/mae-vitb16-saes` | `batchtopk_{8x,16x,32x}_k{64,128,192,256}` |
| DeiT ViT-B/16 | `visaebench/deit-vitb16-saes` | `batchtopk_{8x,16x,32x}_k{64,128,192,256}` |

The 12 subfolders per repo, spelled out:

```
batchtopk_8x_k64    batchtopk_8x_k128    batchtopk_8x_k192    batchtopk_8x_k256
batchtopk_16x_k64   batchtopk_16x_k128   batchtopk_16x_k192   batchtopk_16x_k256
batchtopk_32x_k64   batchtopk_32x_k128   batchtopk_32x_k192   batchtopk_32x_k256
```

Load any config by passing its subfolder:

```python
from visaebench.hub import load_sae

sae = load_sae("visaebench/dinov2-vitb14-saes", subfolder="batchtopk_32x_k256")
```

## Caching

Extraction is the slow part of a run, so activations are cached and reused.

- Shards are written in fp16, `SHARD_SIZE = 5000` images each, as
  `shard_NNN.pt` files, alongside a `meta.json` (backbone, layer,
  num_images, patch_count) and a `norm_stats.pt` (mean and std).
- The cache root defaults to `$VISAEBENCH_CACHE_DIR`, then
  `$XDG_CACHE_HOME/visaebench`, then `~/.cache/visaebench`. Override it
  per call with `cache_dir=`.
- The cache key is `(backbone, layer, source_tag, N)`, where `source_tag`
  encodes whether images came from a local ImageFolder or from streamed
  HuggingFace `imagenet-1k`. On a re-run with a matching key, `evaluate`
  detects the existing shards, verifies the metadata, and skips
  extraction.

To point the cache at a specific directory:

```bash
export VISAEBENCH_CACHE_DIR=/mnt/data/visaebench_cache
```

## Next steps

- [Evaluating your own SAE and custom ViTs](custom_models.md)
- [The 7 metrics in detail](metrics.md)
- Runnable scripts in `examples/`.
