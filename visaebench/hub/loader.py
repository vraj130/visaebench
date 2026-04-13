"""Load SAE checkpoints from HuggingFace Hub or local paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch

from visaebench.hub.sae_wrapper import GenericSAE


def load_sae(
    source: str,
    *,
    device: str = "cuda",
    filename: str = "sae.pt",
    config_filename: str = "config.json",
    revision: str | None = None,
) -> GenericSAE:
    """Load an SAE from HuggingFace Hub or a local directory.

    Args:
        source: Either a HuggingFace repo ID (e.g.
            ``"visaebench/clip_vitb16_batchtopk_16x_k192"``) or a local
            directory path containing ``sae.pt`` and ``config.json``.
        device: Torch device to place the model on.
        filename: Name of the weights file (default ``"sae.pt"``).
        config_filename: Name of the config file (default ``"config.json"``).
        revision: Git revision for HuggingFace downloads (branch, tag, or
            commit hash).

    Returns:
        A :class:`GenericSAE` instance in eval mode on *device*.

    Examples
    --------
    From HuggingFace::

        sae = load_sae("visaebench/clip_vitb16_batchtopk_16x_k192")

    From a local checkpoint::

        sae = load_sae("/data/checkpoints/my_sae/")
    """
    local_path = Path(source)

    if local_path.is_dir():
        weights_path = local_path / filename
        config_path = local_path / config_filename
    else:
        # Download from HuggingFace Hub
        weights_path, config_path = _download_from_hub(
            source, filename, config_filename, revision=revision,
        )

    # ── Load config ──────────────────────────────────────────────────
    config = _load_config(config_path)

    # ── Load weights ─────────────────────────────────────────────────
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # ── Infer dimensions if not in config ────────────────────────────
    config = _infer_dims(config, state_dict)

    # ── Build and return ─────────────────────────────────────────────
    sae = GenericSAE(config, state_dict)
    sae.eval()
    sae.to(device)
    return sae


def load_sae_from_weights(
    state_dict: dict[str, torch.Tensor],
    config: dict,
    *,
    device: str = "cuda",
) -> GenericSAE:
    """Build a GenericSAE directly from a state dict and config.

    Useful when you already have weights in memory (e.g. from the
    ``overcomplete`` library).
    """
    config = _infer_dims(config, state_dict)
    sae = GenericSAE(config, state_dict)
    sae.eval()
    sae.to(device)
    return sae


# ======================================================================
# Internal helpers
# ======================================================================


def _download_from_hub(
    repo_id: str,
    filename: str,
    config_filename: str,
    revision: str | None,
) -> tuple[Path, Path]:
    """Download weights and config from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    weights_path = Path(hf_hub_download(
        repo_id, filename, revision=revision,
    ))
    config_path = Path(hf_hub_download(
        repo_id, config_filename, revision=revision,
    ))
    return weights_path, config_path


def _load_config(path: Path) -> dict:
    """Load config from JSON or YAML."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def _infer_dims(config: dict, state_dict: dict[str, torch.Tensor]) -> dict:
    """Infer input_dim and hidden_dim from weights if not in config."""
    config = dict(config)

    # Try to find encoder weight to infer dimensions
    if "encoder.weight" in state_dict:
        w = state_dict["encoder.weight"]
        config.setdefault("hidden_dim", w.shape[0])
        config.setdefault("input_dim", w.shape[1])
    elif "encoder.final_block.0.weight" in state_dict:
        w = state_dict["encoder.final_block.0.weight"]
        config.setdefault("hidden_dim", w.shape[0])
        config.setdefault("input_dim", w.shape[1])

    # Infer from expansion_factor if available
    if "input_dim" in config and "expansion_factor" in config and "hidden_dim" not in config:
        config["hidden_dim"] = config["input_dim"] * config["expansion_factor"]

    # Default activation
    config.setdefault("activation", config.get("sae_type", config.get("architecture", "topk")))

    return config
