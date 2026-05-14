"""M5: Monosemanticity Score.

Measures whether individual SAE features respond to semantically coherent
stimuli, following Pach et al. (2025).

**Procedure** for each alive SAE feature *j*:

1. Find the top-*k* images with the highest (max-pooled) activation for
   feature *j*.
2. Min-max normalise the activation values to obtain weights in [0, 1].
3. Embed the top-*k* images using a **cross-model** evaluator backbone
   (e.g. if the SAE was trained on CLIP features, use DINOv2 to embed).
4. Compute activation-weighted mean pairwise cosine similarity:

   MS_j = Σ_{a<b} w_a·w_b·cos(e_a, e_b) / Σ_{a<b} w_a·w_b

The overall score is the mean MS across all scored features.

**Why cross-model?**  Using the *same* backbone that produced the SAE's
input features would inflate scores — images that are close in CLIP space
would trivially remain close after reconstruction.  A different backbone
provides an independent semantic similarity signal.

**Interpretation:** higher is better.  A score of 1.0 means every feature's
top-activating images are identical in the evaluator's embedding space.

Reference:
    Pach et al. (2025), "Sparse Autoencoders Learn Monosemantic Features in
    Vision-Language Models".
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from visaebench.core.types import BackboneInterface, MetricResult, SAEInterface
from visaebench.metrics.base import Metric


class MonosemanticityScore(Metric):
    """M5: Monosemanticity Score via cross-model evaluation.

    Parameters
    ----------
    top_k_images:
        Number of top-activating images to compare per feature.
    max_features:
        Maximum number of alive features to score (subsampled randomly
        when exceeded).  This is the main knob for compute vs. precision.
    batch_size:
        Tokens per SAE forward pass during encoding.
    embed_batch_size:
        Images per forward pass through the evaluator backbone.
    """

    name = "monosemanticity"
    dimension = "concept_detection"
    higher_is_better = True

    def __init__(
        self,
        top_k_images: int = 16,
        max_features: int = 1000,
        batch_size: int = 512,
        embed_batch_size: int = 32,
    ) -> None:
        self.top_k_images = top_k_images
        self.max_features = max_features
        self.batch_size = batch_size
        self.embed_batch_size = embed_batch_size

    def compute(
        self,
        sae: SAEInterface,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        device: str = "cuda",
        images: Sequence,
        evaluator_backbone: BackboneInterface,
        mean: torch.Tensor | None = None,
        std: float | None = None,
        seed: int = 42,
        evaluator_layer: int = 11,
        **kwargs,
    ) -> MetricResult:
        """Compute monosemanticity score.

        Args:
            sae: Trained SAE with ``encode`` method.
            activations: Tensor ``[N, P, D]`` or list of shard paths.
            device: Torch device string.
            images: Indexable sequence of PIL images (e.g. a HuggingFace
                dataset column or a list).  Must be aligned with
                *activations* — ``images[i]`` corresponds to the *i*-th
                image in the activation dataset.
            evaluator_backbone: A backbone that satisfies
                :class:`~visaebench.core.types.BackboneInterface`, used to
                embed the top-*k* images.  **Must differ from the backbone
                whose features the SAE was trained on** to avoid circularity.
            mean: Per-dimension mean for normalisation.
            std: Scalar standard deviation for normalisation.
            seed: Random seed for feature subsampling.
            evaluator_layer: Layer index to extract from the evaluator
                backbone (default 11).

        Returns:
            :class:`MetricResult` where ``value`` is the mean MS across
            scored features.  Metadata contains ``ms_std``, ``ms_median``,
            ``ms_distribution`` (list of per-feature scores),
            ``num_features_scored``, and ``num_dead_features``.
        """
        # ── Step 1: Encode all images → [num_images, dict_size] ──────
        image_codes = self._encode_to_image_codes(
            sae, activations, mean=mean, std=std, device=device,
        )
        num_images, dict_size = image_codes.shape

        # ── Step 2: Identify alive features and subsample ────────────
        max_per_feature = np.max(image_codes, axis=0)
        alive_mask = max_per_feature > 0
        alive_indices = np.where(alive_mask)[0]
        num_dead = int((~alive_mask).sum())

        rng = np.random.RandomState(seed)
        if len(alive_indices) > self.max_features:
            sampled = rng.choice(alive_indices, size=self.max_features, replace=False)
            sampled.sort()
        else:
            sampled = alive_indices

        # ── Step 3: For each feature, find top-k image indices ───────
        min_per_feature = np.min(image_codes, axis=0)
        range_per_feature = max_per_feature - min_per_feature
        range_per_feature[range_per_feature == 0] = 1.0

        feature_topk: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        all_needed: set[int] = set()

        for feat_idx in sampled:
            col = image_codes[:, feat_idx]
            k = min(self.top_k_images, num_images)
            topk_idx = np.argpartition(col, -k)[-k:]
            topk_idx = topk_idx[np.argsort(col[topk_idx])[::-1]]

            weights = (col[topk_idx] - min_per_feature[feat_idx]) / range_per_feature[feat_idx]
            feature_topk[feat_idx] = (topk_idx, weights.astype(np.float32))
            all_needed.update(topk_idx.tolist())

        # ── Step 4: Embed needed images via evaluator backbone ───────
        embeddings = self._embed_images(
            sorted(all_needed), images, evaluator_backbone,
            device=device, layer=evaluator_layer,
        )

        # ── Step 5: Compute per-feature MS ───────────────────────────
        ms_scores: list[float] = []

        for feat_idx in sampled:
            topk_idx, weights = feature_topk[feat_idx]
            k = len(topk_idx)

            embeds = np.stack([embeddings[idx] for idx in topk_idx])
            embeds_t = F.normalize(torch.from_numpy(embeds).float(), dim=1)
            weights_t = torch.from_numpy(weights).float()

            # Vectorised weighted pairwise cosine similarity
            sim_matrix = embeds_t @ embeds_t.T  # [k, k]
            weight_matrix = weights_t.unsqueeze(1) * weights_t.unsqueeze(0)

            # Upper triangle only (exclude diagonal)
            mask = torch.triu(torch.ones(k, k, dtype=torch.bool), diagonal=1)
            numer = float((sim_matrix[mask] * weight_matrix[mask]).sum())
            denom = float(weight_matrix[mask].sum())

            if denom > 0:
                ms_scores.append(numer / denom)

        ms_array = np.array(ms_scores, dtype=np.float64)
        mean_ms = float(ms_array.mean()) if len(ms_array) > 0 else 0.0

        return self._make_result(
            value=mean_ms,
            metadata={
                "ms_std": float(ms_array.std()) if len(ms_array) > 0 else 0.0,
                "ms_median": float(np.median(ms_array)) if len(ms_array) > 0 else 0.0,
                "ms_distribution": ms_scores,
                "num_features_scored": len(ms_scores),
                "num_dead_features": num_dead,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_to_image_codes(
        self,
        sae: SAEInterface,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        mean: torch.Tensor | None,
        std: float | None,
        device: str,
    ) -> np.ndarray:
        """Encode activations → max-pooled image codes.

        Returns ``np.ndarray`` of shape ``[num_images, dict_size]``.
        """
        if isinstance(activations, torch.Tensor):
            shards: list = [activations]
            total_images = activations.shape[0]
            probe = activations[0, 0] if activations.ndim == 3 else activations[0]
            probe = probe.unsqueeze(0).float().to(device)
            if mean is not None and std is not None:
                probe = (probe - mean.to(device)) / std
            with torch.no_grad():
                dict_size = sae.encode(probe).shape[-1]
            del probe
        else:
            shards = list(activations)
            total_images = 0
            dict_size = None
            for sp in shards:
                s = torch.load(sp, map_location="cpu", weights_only=True)
                total_images += s.shape[0]
                if dict_size is None:
                    probe = s[0, 0].unsqueeze(0).float().to(device)
                    if mean is not None and std is not None:
                        probe = (probe - mean.to(device)) / std
                    with torch.no_grad():
                        dict_size = sae.encode(probe).shape[-1]
                    del probe
                del s

        tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_path = tmp.name
        tmp.close()

        codes_mmap = np.memmap(
            tmp_path, dtype=np.float32, mode="w+",
            shape=(total_images, dict_size),
        )

        write_idx = 0
        shard_iter = (
            shards
            if isinstance(shards[0], torch.Tensor)
            else [torch.load(sp, map_location="cpu", weights_only=True) for sp in shards]
        )

        with torch.no_grad():
            for shard in shard_iter:
                if shard.ndim == 2:
                    shard = shard.unsqueeze(1)
                N, P, D = shard.shape

                for g_start in range(0, N, 16):
                    g_end = min(g_start + 16, N)
                    group = shard[g_start:g_end].float()
                    G = group.shape[0]

                    if mean is not None and std is not None:
                        group = (group - mean) / std

                    tokens = group.reshape(G * P, D)
                    code_parts: list[torch.Tensor] = []
                    for s in range(0, tokens.shape[0], self.batch_size):
                        batch = tokens[s : s + self.batch_size].to(device)
                        codes = sae.encode(batch)
                        code_parts.append(codes.cpu())
                        del batch, codes

                    all_codes = torch.cat(code_parts).reshape(G, P, -1)
                    pooled = all_codes.max(dim=1).values.numpy()
                    codes_mmap[write_idx : write_idx + G] = pooled
                    write_idx += G

                    del group, tokens, code_parts, all_codes, pooled
                del shard

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        codes_mmap.flush()
        result = np.memmap(
            tmp_path, dtype=np.float32, mode="r",
            shape=(total_images, dict_size),
        )
        import atexit
        atexit.register(lambda: os.unlink(tmp_path))
        return result

    def _embed_images(
        self,
        image_indices: list[int],
        images: Sequence,
        evaluator: BackboneInterface,
        *,
        device: str,
        layer: int,
    ) -> dict[int, np.ndarray]:
        """Embed selected images through the evaluator backbone.

        Returns dict mapping image index → L2-normalised embedding vector.
        """
        embeddings: dict[int, np.ndarray] = {}

        for batch_start in range(0, len(image_indices), self.embed_batch_size):
            batch_idx = image_indices[batch_start : batch_start + self.embed_batch_size]
            pil_images = [images[i] for i in batch_idx]

            with torch.no_grad():
                patch_acts = evaluator.extract_activations(
                    pil_images, layer=layer,
                )
                # Mean-pool patches → [B, D], then L2-normalise
                img_embeds = patch_acts.mean(dim=1)
                img_embeds = F.normalize(img_embeds, dim=1)
                embeds_np = img_embeds.cpu().numpy()

            for j, idx in enumerate(batch_idx):
                embeddings[idx] = embeds_np[j]

            del pil_images, patch_acts, img_embeds

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return embeddings
