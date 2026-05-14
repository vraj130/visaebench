"""M1: Feature Localization Score (Moran's I spatial autocorrelation).

Measures whether individual SAE features activate on **spatially coherent**
regions of the Vision Transformer patch grid.

This is a vision-specific metric with no direct precedent in language-model
SAE benchmarks.  It exploits the fact that ViT patch tokens preserve a
regular 2-D spatial layout, so we can ask: "When this feature fires, do the
active patches cluster together (e.g. an object) or scatter randomly?"

**Method — Moran's I (Moran, 1950)**

For a feature *j* on a single image, let ``x`` be the length-*P* vector of
activation values over the patch grid (reshaped from 1-D token order into a
2-D grid, then flattened back — the ordering is consistent).  Moran's I is:

    I = (N / S) · (x_c^T W x_c) / (x_c^T x_c)

where ``x_c = x − mean(x)``, ``W`` is a binary weight matrix encoding
8-connectivity (queen contiguity) on the grid, ``N`` is the number of
patches, and ``S`` = sum of all entries in ``W``.

- **I ≈ +1**: highly clustered — active patches form a contiguous region
- **I ≈  0**: random spatial arrangement
- **I ≈ −1**: dispersed checkerboard pattern

The per-feature localization score is the mean Moran's I across all valid
images where the feature fires.  The metric-level score is the mean across
all evaluable features.

**Edge-case handling:**

- Features that fire on fewer than ``min_images_per_feature`` images are
  excluded (insufficient evidence).
- Images where a feature fires on fewer than ``min_active_patches`` patches
  are excluded (too few active cells for meaningful spatial analysis).
- Images where a feature fires on more than ``max_active_fraction`` of
  patches are excluded — these are "background" features that activate
  everywhere and would trivially appear spatially coherent.
- Features with zero variance in their activation pattern on a given image
  (all patches have the same value) are excluded for that image.

**Compute controls:**

- ``max_features``: randomly subsample alive features to limit the
  number of features scored (default 500).
- ``max_images_per_feature``: stop accumulating statistics for a feature
  once it has enough valid images (default 200).

**Interpretation:** higher mean Moran's I is better — the SAE has learned
features that correspond to spatially localised visual concepts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import torch

from visaebench.core.types import MetricResult, SAEInterface
from visaebench.metrics.base import Metric


class FeatureLocalization(Metric):
    """M1: Feature localization via Moran's I spatial autocorrelation.

    Parameters
    ----------
    grid_size:
        ``(height, width)`` of the patch grid.  ``(14, 14)`` for 224 px
        images with patch size 16 (CLIP, SigLIP, MAE, DeiT) and also for
        DINOv2 ViT-B/14 (which uses patch size 14 on 224 px → 16×16, but
        ``facebook/dinov2-base`` crops to 14×14 = 196 patches by default;
        use ``(16, 16)`` if your DINOv2 config yields 256 patches).
    min_active_patches:
        Minimum number of non-zero patches for a (feature, image) pair to
        be considered valid.
    max_active_fraction:
        Maximum fraction of patches that may be active.  Images where a
        feature fires on more than this fraction are excluded as
        "background" features.
    min_images_per_feature:
        A feature must have at least this many valid images to be included
        in the final statistics.
    max_features:
        Maximum number of alive features to score.  If more features are
        alive, a random subsample is drawn.
    max_images_per_feature:
        Stop accumulating statistics for a feature once it has this many
        valid images.  Reduces compute on large datasets.
    batch_size:
        Tokens per SAE encoding forward pass.
    image_batch_size:
        Number of images processed simultaneously for the Moran's I
        computation.
    non_localized_threshold:
        Moran's I threshold below which a feature is classified as
        "non-localized".  Used to compute the ``frac_non_localized``
        statistic.
    """

    name = "feature_localization"
    dimension = "spatial_coherence"
    higher_is_better = True

    def __init__(
        self,
        grid_size: tuple[int, int] = (14, 14),
        min_active_patches: int = 5,
        max_active_fraction: float = 0.9,
        min_images_per_feature: int = 50,
        max_features: int = 500,
        max_images_per_feature: int = 200,
        batch_size: int = 512,
        image_batch_size: int = 64,
        non_localized_threshold: float = 0.1,
    ) -> None:
        self.grid_h, self.grid_w = grid_size
        self.min_active_patches = min_active_patches
        self.max_active_fraction = max_active_fraction
        self.min_images_per_feature = min_images_per_feature
        self.max_features = max_features
        self.max_images_per_feature = max_images_per_feature
        self.batch_size = batch_size
        self.image_batch_size = image_batch_size
        self.non_localized_threshold = non_localized_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        sae: SAEInterface,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        device: str = "cuda",
        mean: torch.Tensor | None = None,
        std: float | None = None,
        seed: int = 42,
        **kwargs,
    ) -> MetricResult:
        """Compute the feature localization score.

        Args:
            sae: Trained SAE with ``encode`` method.
            activations: Tensor ``[N, P, D]`` or list of shard file paths,
                where ``P`` must equal ``grid_h × grid_w``.
            device: Torch device string.
            mean: Per-dimension mean for normalisation, shape ``[D]``.
            std: Scalar standard deviation for normalisation.
            seed: Random seed for feature subsampling.

        Returns:
            :class:`MetricResult` where ``value`` is the mean Moran's I
            across evaluable features.  Metadata contains distributional
            statistics, per-feature scores, and the fraction of
            non-localized features.
        """
        num_patches = self.grid_h * self.grid_w
        max_active_count = int(self.max_active_fraction * num_patches)

        # Build spatial weight matrix once
        W = _build_weight_matrix(self.grid_h, self.grid_w).to(device)
        S = float(W.sum())

        # ── Phase 1: Determine dict_size and which features are alive ──
        # We do a quick first-shard pass to discover dict_size, then decide
        # which features to track.
        first_shard = self._load_first_shard(activations)
        assert first_shard.shape[1] == num_patches, (
            f"Shard patch count {first_shard.shape[1]} != expected "
            f"{num_patches} (grid {self.grid_h}×{self.grid_w})"
        )

        dict_size = self._probe_dict_size(sae, first_shard, mean, std, device)

        # Running accumulators over ALL features (cheap — just numpy arrays)
        morans_sum = np.zeros(dict_size, dtype=np.float64)
        valid_count = np.zeros(dict_size, dtype=np.int64)

        # ── Phase 2: Process all shards ──────────────────────────────
        shard_iter = self._iter_shards(activations)

        total_images_processed = 0

        with torch.no_grad():
            for shard in shard_iter:
                N_shard, P, D = shard.shape
                assert P == num_patches

                for g_start in range(0, N_shard, self.image_batch_size):
                    g_end = min(g_start + self.image_batch_size, N_shard)
                    group = shard[g_start:g_end].float()  # [G, P, D]
                    G = group.shape[0]

                    # Normalise
                    if mean is not None and std is not None:
                        group = (group - mean) / std

                    # Encode through SAE in sub-batches
                    tokens = group.reshape(G * P, D)
                    code_parts: list[torch.Tensor] = []
                    for s in range(0, tokens.shape[0], self.batch_size):
                        batch = tokens[s : s + self.batch_size].to(device)
                        codes = sae.encode(batch)
                        code_parts.append(codes)
                        del batch

                    all_codes = torch.cat(code_parts, dim=0)  # [G*P, F]
                    del code_parts
                    all_codes = all_codes.reshape(G, P, -1)  # [G, P, F]

                    # Vectorised Moran's I for all images × all features
                    mi, valid = _compute_morans_i_batch(
                        all_codes, W, num_patches, S,
                        self.min_active_patches, max_active_count,
                    )

                    # Mask features that already have enough valid images
                    capped = valid_count >= self.max_images_per_feature  # [F]
                    if capped.any():
                        cap_mask = torch.from_numpy(capped).to(valid.device)
                        valid = valid & ~cap_mask.unsqueeze(0)  # [G, F]

                    # Accumulate on CPU
                    mi_np = mi.cpu().numpy()       # [G, F]
                    valid_np = valid.cpu().numpy()  # [G, F]
                    mi_np = np.where(valid_np, mi_np, 0.0)
                    morans_sum += mi_np.sum(axis=0)
                    valid_count += valid_np.astype(np.int64).sum(axis=0)

                    total_images_processed += G
                    del all_codes, mi, valid, group, tokens

                del shard

                # Early stop: if all features have hit the image cap
                if np.all(valid_count >= self.max_images_per_feature):
                    break

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # ── Phase 3: Aggregate per-feature scores ────────────────────
        evaluable_mask = valid_count >= self.min_images_per_feature
        per_feature_raw = np.full(dict_size, np.nan, dtype=np.float64)
        per_feature_raw[evaluable_mask] = (
            morans_sum[evaluable_mask] / valid_count[evaluable_mask]
        )

        # Subsample to max_features for final statistics
        evaluable_indices = np.where(evaluable_mask)[0]
        rng = np.random.RandomState(seed)
        if len(evaluable_indices) > self.max_features:
            sampled_indices = rng.choice(
                evaluable_indices, size=self.max_features, replace=False,
            )
            sampled_indices.sort()
        else:
            sampled_indices = evaluable_indices

        sampled_scores = per_feature_raw[sampled_indices]
        num_scored = len(sampled_scores)

        if num_scored > 0:
            mean_mi = float(np.mean(sampled_scores))
            median_mi = float(np.median(sampled_scores))
            std_mi = float(np.std(sampled_scores))
            p25 = float(np.percentile(sampled_scores, 25))
            p75 = float(np.percentile(sampled_scores, 75))
            frac_non_localized = float(
                (sampled_scores < self.non_localized_threshold).mean()
            )
        else:
            mean_mi = 0.0
            median_mi = std_mi = p25 = p75 = 0.0
            frac_non_localized = 1.0

        # Per-feature scores list (None for non-evaluable)
        per_feature_list: list[float | None] = [
            None if np.isnan(v) else float(v) for v in per_feature_raw
        ]

        return self._make_result(
            value=mean_mi,
            metadata={
                "median_morans_i": median_mi,
                "std_morans_i": std_mi,
                "percentile_25": p25,
                "percentile_75": p75,
                "frac_non_localized": frac_non_localized,
                "non_localized_threshold": self.non_localized_threshold,
                "num_features_scored": num_scored,
                "num_evaluable_features": int(evaluable_mask.sum()),
                "num_total_features": dict_size,
                "frac_evaluable": int(evaluable_mask.sum()) / dict_size if dict_size > 0 else 0.0,
                "total_images_processed": total_images_processed,
                "per_feature_scores": per_feature_list,
                "scored_feature_indices": sampled_indices.tolist(),
                "distribution": sampled_scores.tolist(),
                "config": {
                    "grid_size": [self.grid_h, self.grid_w],
                    "connectivity": 8,
                    "min_active_patches": self.min_active_patches,
                    "max_active_fraction": self.max_active_fraction,
                    "min_images_per_feature": self.min_images_per_feature,
                    "max_features": self.max_features,
                    "max_images_per_feature": self.max_images_per_feature,
                },
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _probe_dict_size(
        self,
        sae: SAEInterface,
        shard: torch.Tensor,
        mean: torch.Tensor | None,
        std: float | None,
        device: str,
    ) -> int:
        """Determine SAE dict size from a single token."""
        token = shard[0, 0].unsqueeze(0).float().to(device)
        if mean is not None and std is not None:
            token = (token - mean.to(device)) / std
        with torch.no_grad():
            return sae.encode(token).shape[-1]

    def _load_first_shard(
        self,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
    ) -> torch.Tensor:
        if isinstance(activations, torch.Tensor):
            return activations
        return torch.load(activations[0], map_location="cpu", weights_only=True)

    def _iter_shards(
        self,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
    ):
        """Yield shard tensors of shape [N_i, P, D]."""
        if isinstance(activations, torch.Tensor):
            if activations.ndim == 2:
                raise ValueError(
                    "FeatureLocalization requires 3-D activations [N, P, D] "
                    "to recover the spatial patch grid."
                )
            yield activations
        else:
            for path in activations:
                shard = torch.load(path, map_location="cpu", weights_only=True)
                yield shard
                del shard


# ======================================================================
# Vectorised Moran's I computation
# ======================================================================


def _compute_morans_i_batch(
    codes: torch.Tensor,
    W: torch.Tensor,
    N: int,
    S: float,
    min_active: int,
    max_active: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Moran's I for a batch of images, all features at once.

    Uses a single matrix multiply ``W @ x_c`` (broadcast over the batch and
    feature dims) to compute the spatial autocorrelation numerator for every
    (image, feature) pair simultaneously.  No Python loops over features or
    images.

    Args:
        codes: ``[B, P, F]`` activation codes on device.
        W: ``[P, P]`` spatial weight matrix on device.
        N: Number of patches (``P``).
        S: Sum of all weights in ``W``.
        min_active: Minimum nonzero patches for validity.
        max_active: Maximum nonzero patches for validity (background filter).

    Returns:
        morans_i: ``[B, F]`` Moran's I values (undefined where not valid).
        valid: ``[B, F]`` boolean mask of evaluable entries.
    """
    # Active patches per (image, feature)
    active_count = (codes != 0).sum(dim=1)  # [B, F]

    # Center codes per image: x_c = x - mean(x)
    x_bar = codes.mean(dim=1, keepdim=True)  # [B, 1, F]
    x_c = codes - x_bar                      # [B, P, F]

    # Denominator: sum of squared deviations per (image, feature)
    denom = (x_c ** 2).sum(dim=1)            # [B, F]

    # Numerator: spatial autocorrelation via weight matrix
    # W [P, P] broadcasts across batch dim of x_c [B, P, F]:
    #   (W @ x_c)[b, p, f] = Σ_q W[p, q] · x_c[b, q, f]
    wx_c = torch.matmul(W, x_c)             # [B, P, F]
    numer = (x_c * wx_c).sum(dim=1)         # [B, F]

    # Moran's I = (N/S) · numer / denom
    morans_i = (N / S) * numer / denom.clamp(min=1e-10)

    # Validity mask:
    #   - enough active patches (not too sparse)
    #   - not too many active patches (not background)
    #   - non-zero variance (constant patterns are undefined)
    valid = (
        (active_count >= min_active)
        & (active_count <= max_active)
        & (denom > 1e-10)
    )

    return morans_i, valid


# ======================================================================
# Spatial weight matrix construction
# ======================================================================


def _build_weight_matrix(grid_h: int, grid_w: int) -> torch.Tensor:
    """Build an 8-connectivity (queen contiguity) spatial weight matrix.

    Entry ``W[i, j] = 1`` if patches *i* and *j* are neighbours on the 2-D
    grid (horizontally, vertically, or diagonally adjacent).

    Interior cells have 8 neighbours, edge cells 5, corner cells 3.

    Args:
        grid_h: Grid height (e.g. 14).
        grid_w: Grid width (e.g. 14).

    Returns:
        Dense float32 tensor of shape ``[grid_h * grid_w, grid_h * grid_w]``.
    """
    P = grid_h * grid_w
    W = torch.zeros(P, P, dtype=torch.float32)

    for r in range(grid_h):
        for c in range(grid_w):
            i = r * grid_w + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        j = nr * grid_w + nc
                        W[i, j] = 1.0
    return W
