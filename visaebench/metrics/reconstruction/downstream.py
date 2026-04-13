"""M2: Downstream Classification Preservation.

Measures how much task-relevant information the SAE preserves by comparing
a linear probe's accuracy on original activations versus SAE-reconstructed
activations.

**Procedure:**

1. Pool patch-level activations to image-level representations (mean-pool).
2. Train a logistic regression probe on the original pooled features.
3. Evaluate the *same* probe on SAE-reconstructed pooled features.
4. Report the accuracy gap: ``original_acc − reconstructed_acc``.

An accuracy gap of 0 means the SAE preserves all classification-relevant
information.  Larger gaps indicate information loss.

**Interpretation:** lower ``accuracy_gap`` is better.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch

from visaebench.core.types import MetricResult, SAEInterface
from visaebench.metrics.base import Metric


class DownstreamPreservation(Metric):
    """M2: Downstream classification preservation via linear probing.

    Parameters
    ----------
    test_size:
        Fraction of data used for evaluation (stratified split).
    batch_size:
        Number of tokens per SAE forward pass.
    C:
        Inverse regularisation strength for :class:`~sklearn.linear_model.LogisticRegression`.
    max_iter:
        Maximum solver iterations.
    """

    name = "downstream_preservation"
    dimension = "reconstruction"
    higher_is_better = False

    def __init__(
        self,
        test_size: float = 0.2,
        batch_size: int = 512,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        self.test_size = test_size
        self.batch_size = batch_size
        self.C = C
        self.max_iter = max_iter

    def compute(
        self,
        sae: SAEInterface,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        device: str = "cuda",
        labels: np.ndarray | Sequence[int],
        mean: torch.Tensor | None = None,
        std: float | None = None,
        seed: int = 42,
        **kwargs,
    ) -> MetricResult:
        """Compute downstream preservation.

        Args:
            sae: Trained SAE with ``encode`` / ``decode`` methods.
            activations: A tensor of shape ``[N, P, D]`` (images × patches ×
                hidden_dim) or a list of shard file paths with that shape.
            device: Torch device string.
            labels: Integer class labels, one per image.  Must be aligned
                with the image ordering in *activations*.
            mean: Per-dimension mean for normalisation, shape ``[D]``.
            std: Scalar standard deviation for normalisation.
            seed: Random seed for the train/test split and solver.

        Returns:
            :class:`MetricResult` where ``value`` is the accuracy gap
            (``original_acc − reconstructed_acc``).  Metadata contains
            ``accuracy_original``, ``accuracy_reconstructed``,
            ``preservation_ratio``, and ``num_images``.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        labels = np.asarray(labels, dtype=np.int64)

        # ---- determine d_model and total images from activations -------
        if isinstance(activations, torch.Tensor):
            shards: list[torch.Tensor] = [activations]
        else:
            shards = []  # we'll load lazily
            shard_paths = list(activations)

        # Count total images for memmap allocation
        if isinstance(activations, torch.Tensor):
            total_images = activations.shape[0]
            d_model = activations.shape[-1]
        else:
            total_images = 0
            d_model = None
            for sp in shard_paths:
                s = torch.load(sp, map_location="cpu", weights_only=True)
                total_images += s.shape[0]
                if d_model is None:
                    d_model = s.shape[-1]
                del s

        # ---- create temp memmaps for pooled features -------------------
        tmp_orig = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_recon = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_orig.close()
        tmp_recon.close()

        orig_mmap = np.memmap(
            tmp_orig.name, dtype=np.float32, mode="w+",
            shape=(total_images, d_model),
        )
        recon_mmap = np.memmap(
            tmp_recon.name, dtype=np.float32, mode="w+",
            shape=(total_images, d_model),
        )

        # ---- process shards: encode→decode, mean-pool, store ----------
        write_idx = 0
        shard_iter = shards if shards else (
            torch.load(sp, map_location="cpu", weights_only=True)
            for sp in shard_paths
        )

        with torch.no_grad():
            for shard in shard_iter:
                # shard: [N_i, P, D]
                N = shard.shape[0]
                P = shard.shape[1] if shard.ndim == 3 else 1
                D = shard.shape[-1]

                for g_start in range(0, N, 16):
                    g_end = min(g_start + 16, N)
                    group = shard[g_start:g_end].float()  # [G, P, D] or [G, D]
                    if group.ndim == 2:
                        group = group.unsqueeze(1)
                    G = group.shape[0]

                    # Normalise
                    if mean is not None and std is not None:
                        group = (group - mean) / std

                    # Mean-pool original activations → [G, D]
                    orig_pooled = group.mean(dim=1)

                    # Reconstruct through SAE token-by-token
                    tokens = group.reshape(G * P, D)
                    recon_parts: list[torch.Tensor] = []
                    for s in range(0, tokens.shape[0], self.batch_size):
                        batch = tokens[s : s + self.batch_size].to(device)
                        codes = sae.encode(batch)
                        x_hat = sae.decode(codes)
                        recon_parts.append(x_hat.cpu())
                        del batch, codes, x_hat

                    recon_all = torch.cat(recon_parts).reshape(G, P, D)
                    recon_pooled = recon_all.mean(dim=1)

                    orig_mmap[write_idx : write_idx + G] = orig_pooled.numpy()
                    recon_mmap[write_idx : write_idx + G] = recon_pooled.numpy()
                    write_idx += G

                    del group, tokens, recon_parts, recon_all
                    del orig_pooled, recon_pooled

                del shard

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        orig_mmap.flush()
        recon_mmap.flush()

        # Re-open read-only
        orig_data = np.memmap(
            tmp_orig.name, dtype=np.float32, mode="r",
            shape=(total_images, d_model),
        )
        recon_data = np.memmap(
            tmp_recon.name, dtype=np.float32, mode="r",
            shape=(total_images, d_model),
        )

        # Truncate labels if needed
        if len(labels) > total_images:
            labels = labels[:total_images]

        # ---- stratified split ------------------------------------------
        idx_train, idx_test, y_train, y_test = train_test_split(
            np.arange(total_images),
            labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=seed,
        )

        # ---- fit probe on original, evaluate on both -------------------
        clf = LogisticRegression(
            solver="lbfgs", max_iter=self.max_iter, C=self.C,
            random_state=seed,
        )
        clf.fit(orig_data[idx_train], y_train)
        acc_orig = float(clf.score(orig_data[idx_test], y_test))
        acc_recon = float(clf.score(recon_data[idx_test], y_test))

        # ---- cleanup ---------------------------------------------------
        del orig_data, recon_data, clf
        os.unlink(tmp_orig.name)
        os.unlink(tmp_recon.name)

        gap = acc_orig - acc_recon
        preservation = acc_recon / acc_orig if acc_orig > 0 else 0.0

        return self._make_result(
            value=gap,
            metadata={
                "accuracy_original": acc_orig,
                "accuracy_reconstructed": acc_recon,
                "preservation_ratio": preservation,
                "num_images": total_images,
            },
        )
