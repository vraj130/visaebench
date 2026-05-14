"""M2: Fraction of Variance Unexplained (FVU).

Measures reconstruction quality of a sparse autoencoder by computing the
ratio of residual variance to input variance over held-out activations:

    FVU = Var(x - x̂) / Var(x)

where *x* are the original (normalised) activations and *x̂* = decode(encode(x)).

A perfect reconstruction gives FVU = 0.  Typical SAEs achieve FVU < 0.10.

**Interpretation:** lower is better.  ``explained_variance = 1 - FVU`` is
stored in metadata for convenience.

All statistics are accumulated incrementally (running sums) so that this
metric can process arbitrarily large datasets shard-by-shard without
exceeding memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union

import torch
from torch.utils.data import DataLoader, TensorDataset

from visaebench.core.types import MetricResult, SAEInterface
from visaebench.metrics.base import Metric


class FVU(Metric):
    """Fraction of Variance Unexplained.

    Parameters
    ----------
    batch_size:
        Number of tokens per forward pass through the SAE.
    """

    name = "fvu"
    dimension = "reconstruction"
    higher_is_better = False

    def __init__(self, batch_size: int = 512) -> None:
        self.batch_size = batch_size

    def compute(
        self,
        sae: SAEInterface,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        device: str = "cuda",
        mean: torch.Tensor | None = None,
        std: float | None = None,
        **kwargs,
    ) -> MetricResult:
        """Compute FVU over a set of activations.

        Args:
            sae: Trained SAE with ``encode`` / ``decode`` methods.
            activations: Either a single tensor of shape ``[N, D]`` (already
                flattened patch tokens), or a list of shard file paths where
                each shard is ``[N_i, P, D]`` (images × patches × hidden_dim).
            device: Torch device string.
            mean: Per-dimension mean for normalisation, shape ``[D]``.
                If ``None``, activations are assumed pre-normalised.
            std: Scalar standard deviation for normalisation.
                If ``None``, activations are assumed pre-normalised.

        Returns:
            :class:`MetricResult` with ``value = FVU`` and metadata containing
            ``explained_variance``, ``l0``, ``dead_pct`` (if ``dict_size``
            is provided in *kwargs*).
        """
        dict_size: int | None = kwargs.get("dict_size")

        n_tok = 0
        sum_x = 0.0
        sum_x2 = 0.0
        sum_res = 0.0
        sum_res2 = 0.0
        sum_l0 = 0.0
        ever_active: torch.Tensor | None = None
        if dict_size is not None:
            ever_active = torch.zeros(dict_size, dtype=torch.bool)

        for batch in self._iter_batches(activations, mean=mean, std=std):
            batch = batch.to(device)
            codes = sae.encode(batch)
            x_hat = sae.decode(codes)
            res = batch - x_hat

            n = batch.shape[0]
            n_tok += n
            sum_x += float(batch.sum())
            sum_x2 += float(batch.pow(2).sum())
            sum_res += float(res.sum())
            sum_res2 += float(res.pow(2).sum())
            sum_l0 += float((codes != 0).float().sum())

            if ever_active is not None:
                ever_active |= (codes != 0).any(dim=0).cpu()

            del batch, codes, x_hat, res

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        var_x = sum_x2 / n_tok - (sum_x / n_tok) ** 2
        var_res = sum_res2 / n_tok - (sum_res / n_tok) ** 2
        fvu = float(var_res / var_x) if var_x > 0 else float("inf")

        metadata: dict[str, Any] = {
            "explained_variance": 1.0 - fvu,
            "l0": sum_l0 / n_tok,
            "n_tokens": n_tok,
        }
        if ever_active is not None:
            dead = int((~ever_active).sum())
            metadata["dead_features"] = dead
            metadata["dead_pct"] = 100.0 * dead / dict_size

        return self._make_result(fvu, metadata)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _iter_batches(
        self,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        mean: torch.Tensor | None,
        std: float | None,
    ):
        """Yield batches of shape ``[B, D]`` from *activations*."""
        if isinstance(activations, torch.Tensor):
            yield from self._batches_from_tensor(
                self._normalise(activations, mean, std)
            )
        else:
            # Treat as a sequence of shard file paths
            for path in activations:
                shard = torch.load(path, map_location="cpu", weights_only=True)
                if shard.ndim == 3:
                    N, P, D = shard.shape
                    shard = shard.reshape(N * P, D)
                shard = self._normalise(shard.float(), mean, std)
                yield from self._batches_from_tensor(shard)
                del shard

    def _batches_from_tensor(self, tensor: torch.Tensor):
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=False,
        )
        for (batch,) in loader:
            yield batch

    @staticmethod
    def _normalise(
        x: torch.Tensor,
        mean: torch.Tensor | None,
        std: float | None,
    ) -> torch.Tensor:
        if mean is not None and std is not None:
            return (x - mean) / std
        return x
