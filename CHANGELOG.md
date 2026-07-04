# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-03

Initial public release.

### Added
- Seven evaluation metrics (M1 through M7) spanning four capability dimensions:
  reconstruction, concept detection, spatial coherence, and disentanglement.
- Five ViT-B backbones (CLIP, DINOv2, SigLIP, MAE, DeiT) plus a generic
  `CustomBackbone` adapter for arbitrary HuggingFace or timm ViTs.
- Sixty pretrained SAE checkpoints published on the HuggingFace Hub
  (one repo per backbone, twelve configs each).
- `visaebench.evaluate(...)` single-call entry point with activation caching,
  streaming shard support, and per-metric failure isolation.
- Requires Python 3.10 or newer.

[0.1.0]: https://github.com/vraj130/visaebench/releases/tag/v0.1.0
