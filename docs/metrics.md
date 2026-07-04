# Metrics

VISAEBench reports 7 metrics (M1 to M7) across four capability dimensions.
The runner executes them in the canonical order below, which is also the
default radar-chart sequence. Each metric returns a `MetricResult` tagged
with its dimension, a scalar `value`, a `higher_is_better` flag, and a
`metadata` dict with per-metric detail.

| # | Registry key | Class | Dimension | Higher is better |
|---|---|---|---|---|
| M1 | `localization` | `FeatureLocalization` | spatial_coherence | yes |
| M2 | `fvu` | `FVU` | reconstruction | no |
| M3 | `downstream_preservation` | `DownstreamPreservation` | reconstruction | yes |
| M4 | `sparse_probing` | `SparseProbing` | concept_detection | yes |
| M5 | `monosemanticity` | `MonosemanticityScore` | concept_detection | yes |
| M6 | `cross_domain` | `CrossDomainGeneralization` | concept_detection | yes |
| M7 | `absorption` | `FeatureAbsorption` | disentanglement | no |

Select a subset by registry key, for example
`evaluate(..., metrics=["fvu", "sparse_probing"])`.

---

## M1: Feature Localization (`localization`)

- **Class:** `FeatureLocalization` (name `feature_localization`)
- **Dimension:** spatial_coherence
- **Higher is better:** yes

**What it measures.** Whether individual SAE features fire on spatially
coherent regions of the ViT patch grid. For each feature on each image it
computes Moran's I spatial autocorrelation over the patch activations
using an 8-connectivity (queen contiguity) weight matrix. Moran's I near
+1 means active patches form a contiguous region (like an object), near 0
means a random arrangement, near -1 means a dispersed checkerboard. The
value is the mean Moran's I across evaluable features.

**Key knobs.**

- `grid_size`: `(height, width)` of the patch grid; the shard patch count
  must equal `height x width` (auto-inferred by `evaluate`).
- `min_active_patches` (default 5): minimum non-zero patches for a
  (feature, image) pair to count.
- `max_active_fraction` (default 0.9): images where a feature fires on
  more than this fraction of patches are dropped as "background" features.
- `min_images_per_feature` (default 50): a feature needs this many valid
  images to be scored.
- `max_features` (default 500): random subsample cap for compute.
- `max_images_per_feature` (default 200): stop accumulating once a feature
  has this many valid images.
- `non_localized_threshold` (default 0.1): Moran's I below this counts as
  non-localized for the `frac_non_localized` statistic.

**Interpretation.** Higher mean Moran's I means the SAE has learned
features tied to spatially localised visual concepts.

---

## M2: Fraction of Variance Unexplained (`fvu`)

- **Class:** `FVU`
- **Dimension:** reconstruction
- **Higher is better:** no

**What it measures.** Reconstruction quality, as the ratio of residual
variance to input variance, `FVU = Var(x - x_hat) / Var(x)`, where
`x_hat = decode(encode(x))`. A perfect reconstruction gives 0; typical
SAEs land below 0.10. Statistics are accumulated incrementally so the
metric streams over arbitrarily large datasets. Metadata stores
`explained_variance = 1 - FVU`, mean `l0`, and (when `dict_size` is
passed) dead feature counts.

**Key knobs.**

- `batch_size` (default 512): tokens per SAE forward pass.

**Interpretation.** Lower is better; 0 is a perfect reconstruction.

---

## M3: Downstream Preservation (`downstream_preservation`)

- **Class:** `DownstreamPreservation`
- **Dimension:** reconstruction
- **Higher is better:** yes

**What it measures.** How much task-relevant information survives
reconstruction. It mean-pools patch activations to image level, trains a
logistic-regression probe on the original pooled features, then evaluates
that same probe on SAE-reconstructed pooled features. The value is the
preservation ratio `reconstructed_acc / original_acc`, clamped to [0, 2].

**Key knobs.**

- `test_size` (default 0.2): held-out fraction (stratified split).
- `batch_size` (default 512): tokens per SAE forward pass.
- `C` (default 1.0): inverse regularisation strength for the probe.
- `max_iter` (default 1000): solver iterations.

**Interpretation.** 1.0 means all classification-relevant information is
preserved; below 1.0 indicates information loss.

---

## M4: Sparse Probing (`sparse_probing`)

- **Class:** `SparseProbing` (name `sparse_probing_auc`)
- **Dimension:** concept_detection
- **Higher is better:** yes

**What it measures.** Whether SAE features align with human-recognisable
concepts. It max-pools codes per image, ranks features by ANOVA
F-statistic against class labels, then trains k-sparse logistic probes at
increasing k and records accuracy. The value is the area under the
k-accuracy curve, normalised to [0, 1].

**Key knobs.**

- `k_values` (default `[1, 5, 10, 20, 50, 100]`): sparsity levels
  evaluated; values above the dictionary size are skipped.
- `batch_size` (default 512): tokens per SAE forward pass.
- `test_size` (default 0.2): held-out fraction (stratified split).

**Interpretation.** Higher AUC means individual features are more
predictive of semantic classes, a strong interpretability signal.

---

## M5: Monosemanticity (`monosemanticity`)

- **Class:** `MonosemanticityScore` (name `monosemanticity`)
- **Dimension:** concept_detection
- **Higher is better:** yes

**What it measures.** Whether each feature responds to a semantically
coherent set of stimuli (following Pach et al., 2025). For each alive
feature it takes the top-k activating images, embeds them with a
cross-model evaluator backbone (a different backbone from the one the SAE
was trained on, to avoid circular inflation), and computes the
activation-weighted mean pairwise cosine similarity of those embeddings.
The value is the mean across scored features. `evaluate` supplies the
cross-model evaluator automatically.

**Key knobs.**

- `top_k_images` (default 16): top-activating images compared per feature.
- `max_features` (default 1000): random subsample cap; the main
  compute-versus-precision knob.
- `batch_size` (default 512): tokens per SAE forward pass.
- `embed_batch_size` (default 32): images per evaluator forward pass.

**Interpretation.** Higher is better; 1.0 means a feature's top images are
identical in the evaluator's embedding space.

---

## M6: Cross-Domain Generalization (`cross_domain`)

- **Class:** `CrossDomainGeneralization` (name `cross_domain`)
- **Dimension:** concept_detection
- **Higher is better:** yes

**What it measures.** Whether features learned on ImageNet transfer to
out-of-distribution datasets (EuroSAT and DTD by default). For each OOD
dataset it computes four sub-scores: OOD reconstruction (reported as
1 - FVU), 1 - dead-feature fraction, active-feature Jaccard overlap with
ImageNet, and sparse-probing accuracy at k = 10. The value is the mean of
these four sub-scores across datasets, a single number in [0, 1].

**Key knobs.**

- `k_probe` (default 10): sparsity for the probe sub-score.
- `test_size` (default 0.2): held-out fraction.
- `batch_size` (default 512): tokens per SAE forward pass.
- `backbone_batch_size` (default 32): images per backbone forward pass
  when extracting OOD activations from raw images.

**Interpretation.** Higher is better; a perfectly generalising SAE scores
1.0. `inaturalist` is supported but not a default (its public HF repo
currently 404s); pass it explicitly to use it.

---

## M7: Feature Absorption (`absorption`)

- **Class:** `FeatureAbsorption` (name `absorption_rate`)
- **Dimension:** disentanglement
- **Higher is better:** no

**What it measures.** Absorption, where a broad-category feature swallows a
more specific sub-category so the sub-category has no dedicated feature (a
"dog" feature but no distinct "golden retriever" feature). Using ImageNet's
WordNet sibling groups it computes, per (class, group) pair, a feature
overlap (Jaccard) between the class's top-k features and the group's, plus
an F1-gap between a top-1 probe and a top-4 probe. The value is the mean
absorption rate: the fraction of pairs where absorption is detected. The
shipped `wordnet_pairs.json` (139 sibling groups, 1012 pairs) means NLTK
is not required.

**Key knobs.**

- `min_group_size` (default 3): minimum sibling classes per group.
- `top_k` (default 10): features compared for the overlap signal.
- `absorption_threshold` (default 0.1): F1-gap above which absorption is
  flagged.
- `k_primary` (default 1) and `k_expanded` (default 4): feature counts for
  the narrow and expanded probes in the F1-gap signal.
- `batch_size` (default 512): tokens per SAE forward pass.

**Interpretation.** Lower is better; 0 means every sub-category has its own
dedicated feature(s) distinct from the broader parent.
