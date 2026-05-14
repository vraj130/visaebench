"""Disk-backed activation cache for VISAEBench.

Streams a backbone over an indexable image sequence and writes
fp16 patch-activation shards to ``~/.cache/visaebench/...``.  Re-runs
detect the existing cache by directory naming and skip extraction.

Also computes per-dimension mean and scalar std over the cached
activations in a single streaming pass — required for SAE input
normalisation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm

from visaebench.core.types import BackboneInterface

SHARD_SIZE = 5000  # images per shard (matches the internal repo layout)


def default_cache_dir() -> Path:
    """Return the default activation cache root.

    Honors ``$VISAEBENCH_CACHE_DIR`` if set, otherwise
    ``$XDG_CACHE_HOME/visaebench`` or ``~/.cache/visaebench``.
    """
    env = os.environ.get("VISAEBENCH_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "visaebench"


def extract_and_cache_activations(
    backbone: BackboneInterface,
    backbone_name: str,
    *,
    images,
    layer: int = 11,
    source_tag: str = "imagenet",
    cache_dir: Path | str | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> tuple[list[Path], int]:
    """Extract patch activations from *images* via *backbone*, caching fp16
    shards to disk.

    The cache key is derived from ``(backbone_name, layer, source_tag,
    len(images))``.  If the directory already contains the expected
    shards, extraction is skipped.

    Args:
        backbone: Loaded backbone implementing :class:`BackboneInterface`.
        backbone_name: Registry name, used in the cache directory.
        images: Indexable sequence of PIL images (e.g.
            :class:`visaebench.data.loaders._LazyImages`).
        layer: Transformer block index to extract.
        source_tag: Disambiguates caches from different datasets (e.g.
            ``"imagenet"`` vs ``"imagenet_hf"``).
        cache_dir: Cache root; defaults to :func:`default_cache_dir`.
        batch_size: Images per backbone forward pass.
        show_progress: If True, render a tqdm bar.

    Returns:
        ``(shard_paths, patch_count)``.  *shard_paths* is the ordered
        list of ``shard_*.pt`` files; *patch_count* is the P dimension
        of each shard's ``[N, P, D]`` tensor.
    """
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    n_images = len(images)
    cache_subdir = (
        root
        / "activations"
        / backbone_name
        / f"layer_{layer}"
        / f"{source_tag}_N{n_images}"
    )
    cache_subdir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_subdir / "meta.json"
    expected_shards = (n_images + SHARD_SIZE - 1) // SHARD_SIZE
    existing = sorted(cache_subdir.glob("shard_*.pt"))

    # Cache hit: same n_images, all shards present, meta matches.
    if (
        len(existing) == expected_shards
        and meta_path.is_file()
        and _meta_matches(meta_path, n_images, layer, backbone_name)
    ):
        meta = json.loads(meta_path.read_text())
        return existing, int(meta["patch_count"])

    # Cache miss: extract fresh.
    for stale in existing:
        stale.unlink()

    shard_paths: list[Path] = []
    patch_count: int | None = None

    batches = range(0, n_images, batch_size)
    bar = (
        tqdm(batches, desc=f"extract {backbone_name}/layer_{layer}", leave=False)
        if show_progress
        else batches
    )

    buf: list[torch.Tensor] = []
    buf_count = 0
    shard_idx = 0

    for b in bar:
        batch = [images[i] for i in range(b, min(b + batch_size, n_images))]
        with torch.no_grad():
            acts = backbone.extract_activations(batch, layer=layer)  # [B, P, D]
        acts = acts.detach().to("cpu", dtype=torch.float16)

        if patch_count is None:
            patch_count = int(acts.shape[1])

        buf.append(acts)
        buf_count += acts.shape[0]

        # Flush a shard when we have at least SHARD_SIZE images buffered.
        while buf_count >= SHARD_SIZE:
            shard_paths.append(
                _flush_shard(cache_subdir, shard_idx, buf, SHARD_SIZE)
            )
            shard_idx += 1
            buf_count -= SHARD_SIZE

    # Flush remainder.
    if buf_count > 0:
        shard_paths.append(_flush_shard(cache_subdir, shard_idx, buf, buf_count))

    assert patch_count is not None, "No images were processed."

    meta_path.write_text(json.dumps({
        "backbone": backbone_name,
        "layer": layer,
        "num_images": n_images,
        "patch_count": patch_count,
        "dtype": "float16",
        "shard_size": SHARD_SIZE,
    }, indent=2))

    return shard_paths, patch_count


def compute_or_load_norm_stats(
    shard_paths: Sequence[Path],
    *,
    cache_file: Path | None = None,
) -> tuple[torch.Tensor, float]:
    """Compute per-dimension mean ``[D]`` and scalar std over cached shards.

    Performs a single streaming pass over all shards.  If *cache_file* is
    given and exists, loads the stats from it instead.

    Args:
        shard_paths: List of shard files written by
            :func:`extract_and_cache_activations`.
        cache_file: Optional path for stats persistence.  If supplied,
            stats are loaded from / saved to this file.

    Returns:
        ``(mean, std)`` where *mean* is shape ``[D]`` (float32) and
        *std* is a Python float (scalar RMS over flattened tokens).
    """
    if cache_file is not None and cache_file.is_file():
        blob = torch.load(cache_file, map_location="cpu", weights_only=True)
        return blob["mean"].float(), float(blob["std"])

    n_tokens = 0
    sum_x: torch.Tensor | None = None
    sum_x2: torch.Tensor | None = None

    for sp in shard_paths:
        shard = torch.load(sp, map_location="cpu", weights_only=True)
        if shard.ndim == 3:
            flat = shard.reshape(-1, shard.shape[-1])
        else:
            flat = shard
        flat = flat.float()

        if sum_x is None:
            sum_x = flat.sum(dim=0)
            sum_x2 = (flat * flat).sum(dim=0)
        else:
            sum_x += flat.sum(dim=0)
            sum_x2 += (flat * flat).sum(dim=0)
        n_tokens += flat.shape[0]
        del shard, flat

    assert sum_x is not None and sum_x2 is not None, "No shards provided."
    mean = sum_x / n_tokens
    # Scalar std: RMS of (x − mean) flattened across dims & tokens.
    var_per_dim = sum_x2 / n_tokens - mean * mean
    std = float(var_per_dim.clamp_min(0).mean().sqrt())

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _flush_shard(
    cache_subdir: Path, shard_idx: int, buf: list[torch.Tensor], take: int,
) -> Path:
    """Concatenate buffered tensors, write the first *take* images as a
    shard, and trim the buffer in place."""
    flat = torch.cat(buf, dim=0)
    shard_tensor = flat[:take].contiguous()
    remainder = flat[take:].contiguous()

    path = cache_subdir / f"shard_{shard_idx:03d}.pt"
    torch.save(shard_tensor, path)

    # Reset buffer to just the leftover rows (rebuild list so old refs free).
    buf.clear()
    if remainder.shape[0] > 0:
        buf.append(remainder)
    return path


def _meta_matches(
    meta_path: Path, n_images: int, layer: int, backbone_name: str,
) -> bool:
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(meta.get("num_images", -1)) == n_images
        and int(meta.get("layer", -1)) == layer
        and meta.get("backbone") == backbone_name
    )
