"""GenericSAE: a lightweight SAE wrapper that implements SAEInterface.

Supports three activation functions at inference time:

- **TopK**: keep only the *k* largest activations per sample.
- **BatchTopK**: apply a fixed threshold (``running_threshold`` learned
  during training).  At inference this is equivalent to JumpReLU with a
  single global threshold.
- **JumpReLU**: apply per-concept learned thresholds.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericSAE(nn.Module):
    """Standalone SAE implementing ``encode`` / ``decode``.

    This class wraps raw weight tensors and a config dict into a clean
    inference-only SAE.  It does **not** depend on the ``overcomplete``
    library.

    Parameters
    ----------
    config:
        SAE configuration dict.  Required keys: ``activation``
        (``"topk"`` | ``"batchtopk"`` | ``"jumprelu"``), ``input_dim``,
        ``hidden_dim``.  Optional: ``k``, ``threshold``,
        ``jump_thresholds``.
    state_dict:
        Weight tensors.  Supports two key conventions:

        *Simple format* (our HF upload format)::

            encoder.weight   [hidden_dim, input_dim]
            encoder.bias     [hidden_dim]
            decoder.weight   [hidden_dim, input_dim]
            decoder.bias     [input_dim]        (optional)

        *Overcomplete library format*::

            encoder.final_block.0.weight  [hidden_dim, input_dim]
            encoder.final_block.0.bias    [hidden_dim]
            dictionary._weights           [hidden_dim, input_dim]
            dictionary.multiplier         scalar
    """

    def __init__(self, config: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.config = dict(config)
        activation = self.config.get("activation", self.config.get("sae_type", "topk")).lower()
        self.activation = activation
        self.k = self.config.get("k", None)

        # Infer dimensions from weights if not in config
        self._infer_dims(state_dict)
        self.input_dim = self.config["input_dim"]
        self.hidden_dim = self.config["hidden_dim"]

        # ── Load weights (support both key conventions) ──────────────
        enc_w, enc_b, dec_w, dec_b = _extract_weights(state_dict, self.hidden_dim, self.input_dim)

        self.register_buffer("encoder_weight", enc_w)  # [hidden, input]
        self.register_buffer("encoder_bias", enc_b)     # [hidden]
        self.register_buffer("decoder_weight", dec_w)   # [hidden, input]
        if dec_b is not None:
            self.register_buffer("decoder_bias", dec_b) # [input]
        else:
            self.decoder_bias = None

        # ── Activation-specific parameters ───────────────────────────
        if activation == "batchtopk":
            threshold = config.get("threshold", None)
            if threshold is None and "_running_threshold" in state_dict:
                threshold = float(state_dict["_running_threshold"])
            if threshold is None:
                threshold = 0.0
            self.register_buffer("threshold", torch.tensor(float(threshold)))

        elif activation == "jumprelu":
            if "jump_thresholds" in state_dict:
                jt = state_dict["jump_thresholds"]
            elif "thresholds" in state_dict:
                jt = state_dict["thresholds"]
            else:
                jt = torch.full((self.hidden_dim,), 0.01)
            self.register_buffer("jump_thresholds", jt.float())

    def _infer_dims(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Infer input_dim/hidden_dim from weights if not in config."""
        if "input_dim" not in self.config or "hidden_dim" not in self.config:
            if "encoder.weight" in state_dict:
                w = state_dict["encoder.weight"]
            elif "encoder.final_block.0.weight" in state_dict:
                w = state_dict["encoder.final_block.0.weight"]
            else:
                raise KeyError("Cannot infer dimensions: no encoder weight found")
            self.config.setdefault("hidden_dim", w.shape[0])
            self.config.setdefault("input_dim", w.shape[1])

    # ------------------------------------------------------------------
    # SAEInterface
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations into sparse latent codes.

        Args:
            x: ``[batch, input_dim]``.

        Returns:
            Sparse codes ``[batch, hidden_dim]``.
        """
        # Linear projection + ReLU (matching overcomplete encoder)
        pre_codes = F.relu(x @ self.encoder_weight.T + self.encoder_bias)

        # Sparsify
        if self.activation == "topk":
            return _topk(pre_codes, self.k)
        elif self.activation == "batchtopk":
            return _threshold(pre_codes, self.threshold)
        elif self.activation == "jumprelu":
            return _jumprelu(pre_codes, self.jump_thresholds)
        else:
            raise ValueError(f"Unknown activation: {self.activation!r}")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse codes back to activation space.

        Args:
            z: ``[batch, hidden_dim]``.

        Returns:
            Reconstructed activations ``[batch, input_dim]``.
        """
        x_hat = z @ self.decoder_weight
        if self.decoder_bias is not None:
            x_hat = x_hat + self.decoder_bias
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode.

        Returns:
            ``(codes, x_hat)``
        """
        codes = self.encode(x)
        x_hat = self.decode(codes)
        return codes, x_hat

    def __repr__(self) -> str:
        return (
            f"GenericSAE(activation={self.activation!r}, "
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"k={self.k})"
        )


# ======================================================================
# Activation functions
# ======================================================================


def _topk(codes: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-k values per sample, zero the rest."""
    topk_vals, topk_idx = codes.topk(k, dim=-1)
    sparse = torch.zeros_like(codes)
    sparse.scatter_(-1, topk_idx, topk_vals)
    return sparse


def _threshold(codes: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Zero values below a fixed threshold (BatchTopK at inference)."""
    mask = codes >= threshold
    return codes * mask


def _jumprelu(codes: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Zero values below per-concept thresholds."""
    mask = codes > thresholds.unsqueeze(0)
    return codes * mask


# ======================================================================
# Weight extraction
# ======================================================================


def _extract_weights(
    state_dict: dict[str, torch.Tensor],
    hidden_dim: int,
    input_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Extract encoder/decoder weights from either key convention.

    Returns:
        ``(enc_weight, enc_bias, dec_weight, dec_bias)``
    """
    # Simple format
    if "encoder.weight" in state_dict:
        enc_w = state_dict["encoder.weight"]
        enc_b = state_dict["encoder.bias"]
        dec_w = state_dict["decoder.weight"]
        dec_b = state_dict.get("decoder.bias")
        return enc_w.float(), enc_b.float(), dec_w.float(), dec_b.float() if dec_b is not None else None

    # Overcomplete library format
    if "encoder.final_block.0.weight" in state_dict:
        enc_w = state_dict["encoder.final_block.0.weight"]
        enc_b = state_dict["encoder.final_block.0.bias"]
        dec_w = state_dict["dictionary._weights"]
        # overcomplete dictionary has no bias
        multiplier = state_dict.get("dictionary.multiplier")
        if multiplier is not None and float(multiplier) != 0.0:
            dec_w = dec_w * multiplier.exp()
        return enc_w.float(), enc_b.float(), dec_w.float(), None

    raise KeyError(
        f"Unrecognised state_dict keys: {list(state_dict.keys())}. "
        f"Expected 'encoder.weight' (simple format) or "
        f"'encoder.final_block.0.weight' (overcomplete format)."
    )
