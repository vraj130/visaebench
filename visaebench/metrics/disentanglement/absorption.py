"""M7: Feature Absorption Rate.

Adapted from SAEBench (Karvonen et al., ICML 2025) for the vision domain.

Feature **absorption** occurs when a broad-category SAE feature "absorbs" a
more specific sub-category, preventing the sub-category from being
represented by its own dedicated feature.  For example, if the SAE has a
"dog" feature but no separate "golden retriever" feature, the sub-category
information is absorbed into the broader one.

We test this using ImageNet's WordNet class hierarchy.  Classes are grouped
into **sibling groups** — sets of classes that share the same immediate
WordNet parent (e.g. all terrier breeds, all snake species, all big cats).

**Two complementary signals are computed for each (class, group) pair:**

1. **Feature overlap (primary metric):**
   Find the top-*k* SAE features for the specific class (ranked by mean
   activation on that class's images) and the top-*k* for the whole sibling
   group.  The overlap ratio (Jaccard similarity) measures whether the class
   is represented by a distinct feature set.  *High overlap = absorption.*

2. **F1-gap (secondary signal):**
   Train a binary logistic regression probe (class-vs-siblings) using only
   the top-1 feature, then using the top-4 features.  If F1 improves
   substantially with more features, the single "primary" feature is too
   broad to distinguish the class — another sign of absorption.

The metric value is the **mean absorption rate** across all tested pairs:
the fraction of (class, group) pairs where absorption is detected.

**Interpretation:** lower is better.  A rate of 0 means every sub-category
has its own dedicated feature(s) distinct from the broader parent.

**Hierarchy data:** ships with a precomputed ``wordnet_pairs.json``
(139 sibling groups, 1012 concept pairs) so NLTK is not required.  If NLTK
*is* installed, the metric can build richer groups dynamically.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from visaebench.core.types import MetricResult, SAEInterface
from visaebench.data import IMAGENET_CLASS_INDEX_PATH, WORDNET_PAIRS_PATH
from visaebench.metrics.base import Metric


class FeatureAbsorption(Metric):
    """M7: Feature absorption rate.

    Parameters
    ----------
    min_group_size:
        Minimum number of sibling classes for a group to be included.
    top_k:
        Number of top features to compare for the overlap signal.
    absorption_threshold:
        F1-gap threshold above which absorption is detected.
    k_primary:
        Number of features for the narrow probe (F1-gap signal).
    k_expanded:
        Number of features for the expanded probe (F1-gap signal).
    batch_size:
        Tokens per SAE encoding forward pass.
    """

    name = "absorption_rate"
    dimension = "disentanglement"
    higher_is_better = False

    def __init__(
        self,
        min_group_size: int = 3,
        top_k: int = 10,
        absorption_threshold: float = 0.1,
        k_primary: int = 1,
        k_expanded: int = 4,
        batch_size: int = 512,
    ) -> None:
        self.min_group_size = min_group_size
        self.top_k = top_k
        self.absorption_threshold = absorption_threshold
        self.k_primary = k_primary
        self.k_expanded = k_expanded
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        sibling_groups: dict[str, dict] | None = None,
        **kwargs,
    ) -> MetricResult:
        """Compute the feature absorption rate.

        Args:
            sae: Trained SAE with ``encode`` method.
            activations: Tensor ``[N, P, D]`` or list of shard paths.
            device: Torch device string.
            labels: Integer class labels (ImageNet 0–999), one per image.
            mean: Per-dimension mean for normalisation.
            std: Scalar standard deviation for normalisation.
            seed: Random seed.
            sibling_groups: Optional override for the hierarchy.  If
                ``None``, loads from the shipped ``wordnet_pairs.json``
                (or builds dynamically if NLTK is available).  Format::

                    {"group_name": {"class_indices": [int, ...],
                                    "hierarchy_depth": int}, ...}

        Returns:
            :class:`MetricResult` where ``value`` is the mean absorption
            rate (fraction of pairs showing absorption).  Metadata contains
            per-pair results, F1 statistics, overlap statistics, and
            breakdown by hierarchy depth.
        """
        labels = np.asarray(labels, dtype=np.int64)

        # ── Step 1: Encode all images → [num_images, dict_size] ──────
        image_codes = self._encode_to_image_codes(
            sae, activations, mean=mean, std=std, device=device,
        )
        num_images, dict_size = image_codes.shape

        if num_images < len(labels):
            labels = labels[:num_images]

        # ── Step 2: Load sibling groups ──────────────────────────────
        if sibling_groups is None:
            sibling_groups = load_sibling_groups(
                min_group_size=self.min_group_size,
            )

        # ── Step 3: Per-(group, class) absorption tests ──────────────
        pair_results: list[dict[str, Any]] = []
        absorbed_count = 0

        for group_name, group_info in sibling_groups.items():
            class_indices = group_info["class_indices"]
            depth = group_info.get("hierarchy_depth", 1)

            if len(class_indices) < self.min_group_size:
                continue

            # Images belonging to any class in this group
            group_mask = np.isin(labels, class_indices)
            if group_mask.sum() < 10:
                continue

            group_codes = image_codes[group_mask]
            group_labels = labels[group_mask]

            # Mean activation per feature across ALL group images
            group_mean_act = group_codes.mean(axis=0)  # [dict_size]
            group_top_k = set(np.argsort(group_mean_act)[-self.top_k:])

            for target_class in class_indices:
                binary_labels = (group_labels == target_class).astype(np.int64)
                n_pos = int(binary_labels.sum())
                n_neg = len(binary_labels) - n_pos

                if n_pos < 3 or n_neg < 3:
                    continue

                # ── Feature overlap signal ───────────────────────
                class_mask = group_labels == target_class
                class_codes = group_codes[class_mask]
                class_mean_act = class_codes.mean(axis=0)
                class_top_k = set(np.argsort(class_mean_act)[-self.top_k:])

                intersection = len(class_top_k & group_top_k)
                union = len(class_top_k | group_top_k)
                overlap = intersection / union if union > 0 else 0.0

                # ── F1-gap signal ────────────────────────────────
                f_scores, _ = f_classif(group_codes, binary_labels)
                f_scores = np.nan_to_num(f_scores, nan=-np.inf)
                ranked = np.argsort(f_scores)[::-1]

                f1_k1 = self._train_probe(
                    group_codes, binary_labels, ranked,
                    self.k_primary, seed,
                )
                f1_k4 = self._train_probe(
                    group_codes, binary_labels, ranked,
                    self.k_expanded, seed,
                )
                f1_gap = f1_k4 - f1_k1

                # Absorption detected if EITHER signal triggers
                is_absorbed = (
                    f1_gap > self.absorption_threshold
                    or overlap > 0.5
                )
                if is_absorbed:
                    absorbed_count += 1

                pair_results.append({
                    "group": group_name,
                    "class_idx": int(target_class),
                    "hierarchy_depth": depth,
                    "overlap": overlap,
                    "f1_k1": f1_k1,
                    "f1_k4": f1_k4,
                    "f1_gap": f1_gap,
                    "absorbed": is_absorbed,
                    "n_positive": n_pos,
                })

        # ── Step 4: Aggregate ────────────────────────────────────────
        num_tests = len(pair_results)
        rate = absorbed_count / num_tests if num_tests > 0 else 0.0

        # Per-depth breakdown
        depth_breakdown: dict[int, dict[str, Any]] = {}
        if num_tests > 0:
            for depth in sorted({p["hierarchy_depth"] for p in pair_results}):
                depth_pairs = [p for p in pair_results if p["hierarchy_depth"] == depth]
                n_abs = sum(1 for p in depth_pairs if p["absorbed"])
                depth_breakdown[depth] = {
                    "num_tests": len(depth_pairs),
                    "num_absorbed": n_abs,
                    "absorption_rate": n_abs / len(depth_pairs),
                    "mean_overlap": float(np.mean([p["overlap"] for p in depth_pairs])),
                    "mean_f1_gap": float(np.mean([p["f1_gap"] for p in depth_pairs])),
                }

        # Global statistics
        if num_tests > 0:
            overlaps = np.array([p["overlap"] for p in pair_results])
            f1_k1_arr = np.array([p["f1_k1"] for p in pair_results])
            f1_k4_arr = np.array([p["f1_k4"] for p in pair_results])
            f1_gaps = np.array([p["f1_gap"] for p in pair_results])
        else:
            overlaps = f1_k1_arr = f1_k4_arr = f1_gaps = np.array([0.0])

        return self._make_result(
            value=rate,
            metadata={
                "num_groups": len(sibling_groups),
                "num_tests": num_tests,
                "num_absorbed": absorbed_count,
                "mean_overlap": float(overlaps.mean()),
                "std_overlap": float(overlaps.std()),
                "mean_f1_k1": float(f1_k1_arr.mean()),
                "mean_f1_k4": float(f1_k4_arr.mean()),
                "mean_f1_gap": float(f1_gaps.mean()),
                "depth_breakdown": depth_breakdown,
                "per_pair_results": pair_results,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _train_probe(
        X: np.ndarray,
        y: np.ndarray,
        ranked_features: np.ndarray,
        k: int,
        seed: int,
    ) -> float:
        """Train logistic regression on top-k features, return F1."""
        top_k = ranked_features[:k]
        X_k = X[:, top_k]
        clf = LogisticRegression(
            solver="lbfgs", max_iter=500, C=1.0, random_state=seed,
        )
        clf.fit(X_k, y)
        preds = clf.predict(X_k)
        return float(f1_score(y, preds, average="binary", zero_division=0.0))

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

        # Use memmap for large datasets
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


# ======================================================================
# Sibling group loading (WordNet hierarchy)
# ======================================================================


def load_sibling_groups(
    *,
    min_group_size: int = 3,
) -> dict[str, dict]:
    """Load ImageNet sibling groups from the WordNet hierarchy.

    Tries three sources in order:

    1. **NLTK WordNet** (dynamic, richest) — if ``nltk`` is installed.
    2. **Shipped ``wordnet_pairs.json``** (precomputed, 139 groups).
    3. **Hardcoded fallback** (25 manually curated groups).

    Args:
        min_group_size: Minimum classes per group.

    Returns:
        Dict mapping group name → ``{"class_indices": [...], "hierarchy_depth": int}``.
    """
    # Try NLTK first
    groups = _try_nltk_groups(min_group_size)
    if groups is not None:
        return groups

    # Try shipped JSON
    groups = _try_shipped_json(min_group_size)
    if groups is not None:
        return groups

    # Hardcoded fallback
    return _hardcoded_groups(min_group_size)


def _try_nltk_groups(min_group_size: int) -> dict[str, dict] | None:
    """Build sibling groups from NLTK WordNet."""
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn

        class_index = _load_class_index()
        groups: dict[str, list[int]] = defaultdict(list)

        for idx, (wnid, _) in class_index.items():
            offset = int(wnid[1:])
            try:
                syn = wn.synset_from_pos_and_offset("n", offset)
                hypernyms = syn.hypernyms()
                if hypernyms:
                    groups[hypernyms[0].name()].append(idx)
            except Exception:
                continue

        return {
            k: {"class_indices": sorted(v), "hierarchy_depth": 1}
            for k, v in groups.items()
            if len(v) >= min_group_size
        }
    except (ImportError, LookupError):
        return None


def _try_shipped_json(min_group_size: int) -> dict[str, dict] | None:
    """Load from the precomputed wordnet_pairs.json."""
    if not WORDNET_PAIRS_PATH.exists():
        return None

    with open(WORDNET_PAIRS_PATH) as f:
        data = json.load(f)

    raw_groups = data.get("sibling_groups", {})
    return {
        k: {"class_indices": v["class_indices"], "hierarchy_depth": v.get("hierarchy_depth", 1)}
        for k, v in raw_groups.items()
        if len(v["class_indices"]) >= min_group_size
    }


def _load_class_index() -> dict[int, tuple[str, str]]:
    """Load ImageNet class index → (WNID, readable_name)."""
    with open(IMAGENET_CLASS_INDEX_PATH) as f:
        data = json.load(f)
    return {int(k): (v[0], v[1]) for k, v in data.items()}


def _hardcoded_groups(min_group_size: int) -> dict[str, dict]:
    """Fallback: manually curated ImageNet superclass groups.

    Covers ~25 diverse categories spanning animals, vehicles, food,
    instruments, and everyday objects.
    """
    raw = {
        "terrier": list(range(181, 199)),
        "hound": list(range(160, 172)),
        "shepherd_dog": list(range(226, 236)),
        "toy_dog": list(range(151, 157)),
        "sporting_dog": list(range(206, 216)),
        "working_dog": list(range(243, 253)),
        "big_cat": [288, 289, 290, 291, 292, 293],
        "bear": [294, 295, 296, 297],
        "primate": list(range(365, 386)),
        "snake": list(range(52, 69)),
        "spider": [72, 73, 74, 75, 76, 77],
        "beetle": [300, 301, 302, 303, 304, 305, 306],
        "butterfly": [320, 321, 322, 323, 324],
        "fish": [0, 1, 2, 389, 390, 391, 392, 393, 394, 395, 396, 397],
        "bird_wading": list(range(129, 146)),
        "vehicle_wheeled": [407, 436, 468, 511, 609, 627, 654, 656, 661, 671, 675, 705, 717, 734, 751, 779, 817, 864],
        "boat": [427, 435, 463, 472, 484, 554, 625],
        "furniture_seating": [423, 559, 765, 831, 857],
        "musical_instrument": [401, 402, 420, 431, 432, 486, 494, 513, 541, 546, 558, 566, 579, 593, 642, 687, 776, 822, 875, 889],
        "food_fruit": list(range(948, 958)),
        "food_vegetable": list(range(937, 948)),
        "kitchen_utensil": [462, 499, 567, 700, 868, 910, 963],
        "container_bottle": [440, 720, 737, 898, 907],
        "ball": [429, 430, 522, 574, 722, 747, 768, 805],
        "screen_display": [527, 664, 681, 782, 851],
        "clothing": [474, 514, 617, 638, 639, 640, 689, 834],
    }
    return {
        k: {"class_indices": sorted(v), "hierarchy_depth": 1}
        for k, v in raw.items()
        if len(v) >= min_group_size
    }
