"""M3: Sparse Probing Accuracy Curve.

Measures whether SAE features align with human-recognisable concepts by
training *k*-sparse linear probes at increasing sparsity levels.

**Procedure** (for each *k* in the schedule):

1. Encode all images through the SAE and max-pool codes across patches to
   obtain one code vector per image.
2. Rank features by their F-statistic (ANOVA) with respect to class labels
   — this is a fast, non-parametric proxy for mutual information.
3. Select the top-*k* features and train a logistic regression probe.
4. Record the probe's classification accuracy.

The primary metric is the **area under the k-accuracy curve** (AUC),
normalised to [0, 1].  Higher AUC means the SAE learns features that are
individually predictive of semantic classes, which is a strong signal of
interpretability.

**Interpretation:** higher is better.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from visaebench.core.types import MetricResult, SAEInterface
from visaebench.metrics.base import Metric

DEFAULT_K_VALUES = [1, 5, 10, 20, 50, 100]


class SparseProbing(Metric):
    """M3: Sparse probing accuracy curve.

    Parameters
    ----------
    k_values:
        Sparsity levels to evaluate.  Defaults to ``[1, 5, 10, 20, 50, 100]``.
    batch_size:
        Tokens per SAE forward pass.
    test_size:
        Fraction held out for evaluation (stratified split).
    """

    name = "sparse_probing_auc"
    dimension = "concept_detection"
    higher_is_better = True

    def __init__(
        self,
        k_values: list[int] | None = None,
        batch_size: int = 512,
        test_size: float = 0.2,
    ) -> None:
        self.k_values = k_values or list(DEFAULT_K_VALUES)
        self.batch_size = batch_size
        self.test_size = test_size

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
        """Compute sparse probing AUC.

        Args:
            sae: Trained SAE with ``encode`` / ``decode`` methods.
            activations: Tensor ``[N, P, D]`` or list of shard paths with
                that shape.  Patch-level activations are max-pooled across
                patches after encoding to yield image-level code vectors.
            device: Torch device string.
            labels: Integer class labels, one per image.
            mean: Per-dimension mean for normalisation, shape ``[D]``.
            std: Scalar standard deviation for normalisation.
            seed: Random seed for splits and solver.

        Returns:
            :class:`MetricResult` where ``value`` is the normalised AUC.
            Metadata contains ``k_accuracies`` (dict mapping each *k* to
            its accuracy) and ``num_images``.
        """
        labels = np.asarray(labels, dtype=np.int64)

        # Step 1: Encode → max-pool → [num_images, dict_size]
        image_codes = self._encode_to_image_codes(
            sae, activations, mean=mean, std=std, device=device,
        )
        num_images, dict_size = image_codes.shape

        if num_images < len(labels):
            labels = labels[:num_images]

        # Step 2: Feature ranking by F-statistic
        f_scores, _ = f_classif(image_codes, labels)
        f_scores = np.nan_to_num(f_scores, nan=-np.inf)
        ranked = np.argsort(f_scores)[::-1]

        # Step 3: Stratified train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            image_codes, labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=seed,
        )

        # Step 4: k-sparse probes
        # Filter k values that exceed available features
        valid_ks = [k for k in sorted(self.k_values) if k <= dict_size]
        k_accuracies: dict[int, float] = {}

        for k in valid_ks:
            top_k = ranked[:k]
            clf = LogisticRegression(
                solver="lbfgs", max_iter=500, C=1.0, random_state=seed,
            )
            clf.fit(X_train[:, top_k], y_train)
            k_accuracies[k] = float(clf.score(X_test[:, top_k], y_test))

        # Step 5: AUC (trapezoidal, normalised)
        ks = sorted(k_accuracies)
        accs = [k_accuracies[k] for k in ks]
        if len(ks) > 1:
            auc = float(np.trapezoid(accs, x=ks) / (ks[-1] - ks[0]))
        else:
            auc = accs[0] if accs else 0.0

        return self._make_result(
            value=auc,
            metadata={
                "k_accuracies": k_accuracies,
                "num_images": num_images,
                "dict_size": dict_size,
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
        """Encode activations → max-pooled image codes via memmap.

        Returns:
            ``np.ndarray`` of shape ``[num_images, dict_size]``.
        """
        # Discover shapes
        if isinstance(activations, torch.Tensor):
            shard_list: list = [activations]
            total_images = activations.shape[0]
            # Infer dict_size from a probe encode
            probe = activations[0, 0] if activations.ndim == 3 else activations[0]
            probe = probe.unsqueeze(0).float().to(device)
            if mean is not None and std is not None:
                probe = (probe - mean.to(device)) / std
            with torch.no_grad():
                dict_size = sae.encode(probe).shape[-1]
            del probe
        else:
            shard_list = list(activations)
            total_images = 0
            dict_size = None
            for sp in shard_list:
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

        # Allocate memmap
        tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_path = tmp.name
        tmp.close()

        codes_mmap = np.memmap(
            tmp_path, dtype=np.float32, mode="w+",
            shape=(total_images, dict_size),
        )

        write_idx = 0
        shard_iter = (
            shard_list
            if isinstance(shard_list[0], torch.Tensor)
            else [torch.load(sp, map_location="cpu", weights_only=True) for sp in shard_list]
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
        # Schedule cleanup
        import atexit
        atexit.register(lambda: os.unlink(tmp_path))
        return result
