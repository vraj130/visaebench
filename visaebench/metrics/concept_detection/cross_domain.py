"""M6: Cross-Domain Generalization.

Tests whether SAE features learned on ImageNet transfer to out-of-distribution
(OOD) datasets.  Good features should capture general visual concepts that
generalise beyond the training domain.

For each OOD dataset the metric computes four sub-scores:

a) **OOD reconstruction FVU** — does the SAE still reconstruct well on
   unseen domains?  (lower is better → we report 1 − FVU for the composite)
b) **Dead feature fraction** — how many ImageNet-active features never fire
   on this domain?  (lower is better → we report 1 − dead_frac)
c) **Active feature overlap** — Jaccard similarity between the sets of
   features active on ImageNet vs. on the OOD domain.  (higher is better)
d) **Sparse probing accuracy at k = 10** — can a 10-sparse probe on SAE
   features classify the OOD dataset?  (higher is better)

The primary ``value`` is the mean of these four sub-scores across all OOD
datasets, giving a single number in [0, 1] where **higher is better**.
Per-dataset breakdowns are stored in ``metadata``.

**Interpretation:** higher composite score is better.  An SAE with
perfect cross-domain generalisation would score 1.0.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from visaebench.core.types import BackboneInterface, MetricResult, SAEInterface
from visaebench.metrics.base import Metric

# Type alias for the value side of the ood_datasets dict.
# Each dataset can be supplied in one of two ways:
#   1. Pre-extracted: {"activations": Tensor|list[path], "labels": array}
#   2. From images:   {"images": Sequence[PIL.Image], "labels": array}
#      (requires the `backbone` kwarg so activations can be extracted)
OODEntry = dict[str, Any]


class CrossDomainGeneralization(Metric):
    """M6: Cross-domain generalization.

    Parameters
    ----------
    k_probe:
        Sparsity level for the sparse probe (sub-score d).
    test_size:
        Fraction held out for evaluation.
    batch_size:
        Tokens per SAE forward pass.
    backbone_batch_size:
        Images per backbone forward pass when extracting OOD activations
        from raw images.

    Notes
    -----
    ``inaturalist`` is supported but no longer part of the default
    ``DEFAULT_OOD_DATASETS`` (the public ``huggan/inat_mini`` HF repo
    currently 404s). To use it, pass
    ``ood_datasets={"inaturalist": <local_path_or_hf_repo>}`` to
    :func:`visaebench.evaluate`.
    """

    name = "cross_domain"
    dimension = "concept_detection"
    higher_is_better = True

    def __init__(
        self,
        k_probe: int = 10,
        test_size: float = 0.2,
        batch_size: int = 512,
        backbone_batch_size: int = 32,
    ) -> None:
        self.k_probe = k_probe
        self.test_size = test_size
        self.batch_size = batch_size
        self.backbone_batch_size = backbone_batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        sae: SAEInterface,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        device: str = "cuda",
        ood_datasets: dict[str, OODEntry],
        mean: torch.Tensor | None = None,
        std: float | None = None,
        backbone: BackboneInterface | None = None,
        backbone_layer: int = 11,
        seed: int = 42,
        **kwargs,
    ) -> MetricResult:
        """Compute cross-domain generalization.

        Args:
            sae: Frozen SAE trained on ImageNet activations.
            activations: **ImageNet** activations — tensor ``[N, P, D]`` or
                list of shard paths.  Used to determine which features are
                active in-distribution (needed for the Jaccard overlap and
                dead-feature sub-scores).
            device: Torch device string.
            ood_datasets: Mapping from dataset name to a dict with either:

                * ``{"activations": Tensor|list[path], "labels": array}``
                  — pre-extracted OOD patch activations.
                * ``{"images": Sequence[PIL.Image], "labels": array}``
                  — raw images; requires *backbone* to extract activations.
            mean: Per-dimension mean for normalisation (ImageNet stats).
            std: Scalar standard deviation for normalisation.
            backbone: A :class:`BackboneInterface` used to extract OOD
                activations when *images* are provided instead of
                pre-extracted activations.
            backbone_layer: Layer index for backbone extraction.
            seed: Random seed.

        Returns:
            :class:`MetricResult` with the composite score and per-dataset
            breakdowns in metadata.
        """
        # ── Step 1: Compute ImageNet active feature set ──────────────
        id_codes = self._encode_to_image_codes(
            sae, activations, mean=mean, std=std, device=device,
        )
        id_active = np.max(id_codes, axis=0) > 0  # [dict_size] bool mask
        dict_size = id_codes.shape[1]
        del id_codes

        # ── Step 2: Evaluate each OOD dataset ────────────────────────
        per_dataset: dict[str, dict[str, Any]] = {}

        for ds_name, entry in ood_datasets.items():
            labels = np.asarray(entry["labels"], dtype=np.int64)

            # Get OOD activations — either pre-extracted or from images
            if "activations" in entry:
                ood_acts = entry["activations"]
            elif "images" in entry:
                if backbone is None:
                    raise ValueError(
                        f"OOD dataset {ds_name!r} provides images but no "
                        f"backbone was passed to extract activations."
                    )
                ood_acts = self._extract_activations(
                    entry["images"], backbone, device=device,
                    layer=backbone_layer,
                )
            else:
                raise ValueError(
                    f"OOD entry {ds_name!r} must contain 'activations' or 'images'."
                )

            per_dataset[ds_name] = self._evaluate_one_dataset(
                sae=sae,
                ood_activations=ood_acts,
                labels=labels,
                id_active=id_active,
                dict_size=dict_size,
                mean=mean,
                std=std,
                device=device,
                seed=seed,
            )

        # ── Step 3: Composite score ──────────────────────────────────
        all_composites = [d["composite"] for d in per_dataset.values()]
        composite = float(np.mean(all_composites)) if all_composites else 0.0

        return self._make_result(
            value=composite,
            metadata={"per_dataset": per_dataset},
        )

    # ------------------------------------------------------------------
    # Per-dataset evaluation
    # ------------------------------------------------------------------

    def _evaluate_one_dataset(
        self,
        sae: SAEInterface,
        ood_activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        labels: np.ndarray,
        id_active: np.ndarray,
        dict_size: int,
        mean: torch.Tensor | None,
        std: float | None,
        device: str,
        seed: int,
    ) -> dict[str, Any]:
        """Run all four sub-scores on one OOD dataset.

        Returns a dict with individual sub-scores and the composite.
        """
        # Encode OOD activations → [num_images, dict_size]
        ood_codes = self._encode_to_image_codes(
            sae, ood_activations, mean=mean, std=std, device=device,
        )
        num_images = ood_codes.shape[0]

        # Also compute raw (normalised, mean-pooled) for FVU baseline
        ood_fvu = self._compute_ood_fvu(
            sae, ood_activations, mean=mean, std=std, device=device,
        )

        # (b) Dead feature fraction: features active on ImageNet but not on OOD
        ood_active = np.max(ood_codes, axis=0) > 0
        id_active_count = int(id_active.sum())
        # Of the features active on ImageNet, how many are dead on OOD?
        dead_on_ood = id_active & ~ood_active
        dead_frac = float(dead_on_ood.sum() / id_active_count) if id_active_count > 0 else 0.0

        # (c) Jaccard overlap
        intersection = int((id_active & ood_active).sum())
        union = int((id_active | ood_active).sum())
        jaccard = float(intersection / union) if union > 0 else 0.0

        # (d) Sparse probing at k=k_probe
        if len(labels) > num_images:
            labels = labels[:num_images]

        probe_acc = self._sparse_probe(
            ood_codes, labels, k=self.k_probe, seed=seed,
        )

        # Clamp ood_fvu to [0, 1] for the composite: values outside this
        # range come from numerical edge cases (var_x ≈ 0 → fvu=inf) and
        # would otherwise propagate -inf through the mean.
        recon_quality = max(0.0, min(1.0, 1.0 - ood_fvu))

        sub_scores = {
            "ood_fvu": ood_fvu,
            "reconstruction_quality": recon_quality,
            "dead_feature_fraction": dead_frac,
            "domain_coverage": 1.0 - dead_frac,
            "jaccard_overlap": jaccard,
            f"probe_accuracy_k{self.k_probe}": probe_acc,
            "num_images": num_images,
            "num_classes": int(len(np.unique(labels))),
            "id_active_features": id_active_count,
            "ood_active_features": int(ood_active.sum()),
        }

        sub_scores["composite"] = float(np.mean([
            recon_quality,          # (a) reconstruction quality (clamped)
            1.0 - dead_frac,        # (b) domain coverage
            jaccard,                # (c) feature overlap
            probe_acc,              # (d) sparse probing
        ]))

        return sub_scores

    # ------------------------------------------------------------------
    # Sub-score computations
    # ------------------------------------------------------------------

    def _compute_ood_fvu(
        self,
        sae: SAEInterface,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        mean: torch.Tensor | None,
        std: float | None,
        device: str,
    ) -> float:
        """Compute FVU on OOD activations.

        Uses fp64 accumulators to avoid the catastrophic cancellation that
        ``Var(x) = E[x²] − E[x]²`` suffers when ``Var(x)`` is small relative
        to the squared mean (which is common for normalised OOD activations).
        """
        n_tok = 0
        sum_x = torch.zeros(1, dtype=torch.float64)
        sum_x2 = torch.zeros(1, dtype=torch.float64)
        sum_res = torch.zeros(1, dtype=torch.float64)
        sum_res2 = torch.zeros(1, dtype=torch.float64)

        with torch.no_grad():
            for batch in self._iter_token_batches(activations, mean=mean, std=std):
                batch = batch.to(device)
                codes = sae.encode(batch)
                x_hat = sae.decode(codes)
                res = batch - x_hat

                n = batch.shape[0]
                n_tok += n
                sum_x += batch.double().sum().cpu()
                sum_x2 += batch.double().pow(2).sum().cpu()
                sum_res += res.double().sum().cpu()
                sum_res2 += res.double().pow(2).sum().cpu()

                del batch, codes, x_hat, res

        if n_tok == 0:
            return 1.0

        var_x = (sum_x2 / n_tok - (sum_x / n_tok) ** 2).item()
        var_res = (sum_res2 / n_tok - (sum_res / n_tok) ** 2).item()
        if var_x <= 0:
            return 1.0  # degenerate input — treat as worst-case reconstruction
        return float(max(0.0, var_res) / var_x)

    def _sparse_probe(
        self,
        codes: np.ndarray,
        labels: np.ndarray,
        *,
        k: int,
        seed: int,
    ) -> float:
        """Train a k-sparse logistic regression probe and return accuracy."""
        stratify = labels if np.min(np.bincount(labels)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            codes, labels,
            test_size=self.test_size,
            stratify=stratify,
            random_state=seed,
        )

        # Feature ranking
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            f_scores, _ = f_classif(X_train, y_train)
        f_scores = np.nan_to_num(f_scores, nan=-np.inf)
        top_k = np.argsort(f_scores)[::-1][:k]

        clf = LogisticRegression(
            solver="lbfgs", max_iter=500, C=1.0, random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            clf.fit(X_train[:, top_k], y_train)
        return float(clf.score(X_test[:, top_k], y_test))

    # ------------------------------------------------------------------
    # Activation extraction and encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_activations(
        self,
        images: Sequence,
        backbone: BackboneInterface,
        *,
        device: str,
        layer: int,
    ) -> torch.Tensor:
        """Extract patch activations from raw images via backbone.

        Returns tensor ``[N, P, D]`` on CPU.
        """
        parts: list[torch.Tensor] = []
        for i in range(0, len(images), self.backbone_batch_size):
            batch_imgs = list(images[i : i + self.backbone_batch_size])
            acts = backbone.extract_activations(batch_imgs, layer=layer)
            parts.append(acts.cpu())
            del acts

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return torch.cat(parts, dim=0)

    def _encode_to_image_codes(
        self,
        sae: SAEInterface,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        mean: torch.Tensor | None,
        std: float | None,
        device: str,
    ) -> np.ndarray:
        """Encode activations → max-pooled image-level SAE codes.

        Returns ``np.ndarray`` of shape ``[num_images, dict_size]``.
        """
        # Materialise shards if paths
        if isinstance(activations, torch.Tensor):
            shards = [activations]
        else:
            shards = [
                torch.load(p, map_location="cpu", weights_only=True)
                for p in activations
            ]

        # Probe dict_size
        probe = shards[0][0, 0] if shards[0].ndim == 3 else shards[0][0]
        probe = probe.unsqueeze(0).float().to(device)
        if mean is not None and std is not None:
            probe = (probe - mean.to(device)) / std
        with torch.no_grad():
            dict_size = sae.encode(probe).shape[-1]
        del probe

        total_images = sum(s.shape[0] for s in shards)
        all_codes = np.zeros((total_images, dict_size), dtype=np.float32)
        write_idx = 0

        with torch.no_grad():
            for shard in shards:
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

                    group_codes = torch.cat(code_parts).reshape(G, P, -1)
                    pooled = group_codes.max(dim=1).values.numpy()
                    all_codes[write_idx : write_idx + G] = pooled
                    write_idx += G

                    del group, tokens, code_parts, group_codes, pooled

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return all_codes

    def _iter_token_batches(
        self,
        activations: Union[torch.Tensor, Sequence[Union[str, Path]]],
        *,
        mean: torch.Tensor | None,
        std: float | None,
    ):
        """Yield batches of flattened tokens ``[B, D]``."""
        if isinstance(activations, torch.Tensor):
            shards = [activations]
        else:
            shards = [
                torch.load(p, map_location="cpu", weights_only=True)
                for p in activations
            ]

        for shard in shards:
            if shard.ndim == 3:
                N, P, D = shard.shape
                flat = shard.reshape(N * P, D).float()
            else:
                flat = shard.float()

            if mean is not None and std is not None:
                flat = (flat - mean) / std

            for s in range(0, flat.shape[0], self.batch_size):
                yield flat[s : s + self.batch_size]

            del shard, flat


# ======================================================================
# OOD dataset loaders (convenience helpers)
# ======================================================================


def load_ood_dataset(
    name: str,
    *,
    max_images: int = 10_000,
    seed: int = 42,
) -> tuple[list, np.ndarray]:
    """Download and return ``(images, labels)`` for a supported OOD dataset.

    Supported names: ``"eurosat"``, ``"inaturalist"``, ``"dtd"``.

    Args:
        name: Dataset identifier.
        max_images: Cap on the number of images returned.
        seed: Random seed for subsampling when ``max_images < len(dataset)``.

    Returns:
        Tuple of ``(images, labels)`` where *images* is a list of
        PIL images and *labels* is an ``np.ndarray`` of integer class indices.
    """
    loaders = {
        "eurosat": _load_eurosat,
        "inaturalist": _load_inaturalist,
        "dtd": _load_dtd,
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown OOD dataset {name!r}. Supported: {sorted(loaders)}"
        )
    return loaders[name](max_images=max_images, seed=seed)


def _load_eurosat(
    max_images: int, seed: int,
) -> tuple[list, np.ndarray]:
    """Load EuroSAT satellite imagery (10 classes) via HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("tanganke/eurosat", split="train")
    indices = _subsample_indices(len(ds), max_images, seed)
    images = [ds[int(i)]["image"].convert("RGB") for i in indices]
    labels = np.array([ds[int(i)]["label"] for i in indices], dtype=np.int64)
    return images, labels


def _load_inaturalist(
    max_images: int, seed: int,
) -> tuple[list, np.ndarray]:
    """Load iNaturalist 2021 Mini via HuggingFace datasets.

    Uses the ``"kingdom"`` field as the classification target (coarse-grained
    taxonomy with ~10 classes), keeping the evaluation tractable.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "huggan/inat_mini", split="train",
    )

    # Build label mapping from the "kingdom" or "label" column
    if "kingdom" in ds.column_names:
        raw_labels = ds["kingdom"]
    elif "label" in ds.column_names:
        raw_labels = ds["label"]
    else:
        # Fallback: use whatever classification column exists
        label_col = [c for c in ds.column_names if c not in ("image", "file_name")]
        if not label_col:
            raise RuntimeError("Cannot find label column in iNaturalist dataset")
        raw_labels = ds[label_col[0]]

    # Map string labels to integers if needed
    if isinstance(raw_labels[0], str):
        unique = sorted(set(raw_labels))
        label_map = {v: i for i, v in enumerate(unique)}
        int_labels = [label_map[v] for v in raw_labels]
    else:
        int_labels = list(raw_labels)

    indices = _subsample_indices(len(ds), max_images, seed)
    images = [ds[int(i)]["image"].convert("RGB") for i in indices]
    labels = np.array([int_labels[int(i)] for i in indices], dtype=np.int64)
    return images, labels


def _load_dtd(
    max_images: int, seed: int,
) -> tuple[list, np.ndarray]:
    """Load Describable Textures Dataset (47 classes) via torchvision."""
    try:
        from torchvision.datasets import DTD as DTDDataset

        ds = DTDDataset(root="/tmp/visaebench_dtd", split="train", download=True)
        indices = _subsample_indices(len(ds), max_images, seed)
        images = [ds[int(i)][0].convert("RGB") for i in indices]
        labels = np.array([ds[int(i)][1] for i in indices], dtype=np.int64)
        return images, labels
    except Exception:
        # Fallback to HuggingFace
        from datasets import load_dataset

        ds = load_dataset("tanganke/dtd", split="train")
        indices = _subsample_indices(len(ds), max_images, seed)
        images = [ds[int(i)]["image"].convert("RGB") for i in indices]
        labels = np.array([ds[int(i)]["label"] for i in indices], dtype=np.int64)
        return images, labels


def _subsample_indices(total: int, max_n: int, seed: int) -> np.ndarray:
    """Return up to *max_n* indices, randomly subsampled if needed."""
    if total <= max_n:
        return np.arange(total)
    rng = np.random.RandomState(seed)
    return rng.choice(total, size=max_n, replace=False)
