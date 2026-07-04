"""CPU-only tests for GenericSAE and the SAE loader (no network, no GPU)."""

from __future__ import annotations

import json

import torch

from visaebench.core.types import SAEInterface
from visaebench.hub.loader import load_sae, load_sae_from_weights
from visaebench.hub.sae_wrapper import GenericSAE

INPUT_DIM = 16
HIDDEN_DIM = 64
BATCH = 8


def _simple_state_dict(*, with_decoder_bias: bool = True):
    """Build a simple-format state dict with small random weights."""
    torch.manual_seed(0)
    sd = {
        "encoder.weight": torch.randn(HIDDEN_DIM, INPUT_DIM),   # [hidden, input]
        "encoder.bias": torch.randn(HIDDEN_DIM),                # [hidden]
        "decoder.weight": torch.randn(HIDDEN_DIM, INPUT_DIM),   # [hidden, input]
    }
    if with_decoder_bias:
        sd["decoder.bias"] = torch.randn(INPUT_DIM)            # [input]
    return sd


def _overcomplete_state_dict():
    """Build an overcomplete-format state dict with small random weights."""
    torch.manual_seed(1)
    return {
        "encoder.final_block.0.weight": torch.randn(HIDDEN_DIM, INPUT_DIM),
        "encoder.final_block.0.bias": torch.randn(HIDDEN_DIM),
        "dictionary._weights": torch.randn(HIDDEN_DIM, INPUT_DIM),
        "dictionary.multiplier": torch.tensor(0.0),  # zero => no scaling
    }


def _x():
    torch.manual_seed(2)
    return torch.randn(BATCH, INPUT_DIM)


# ---------------------------------------------------------------------------
# Activation variants: shapes + finiteness
# ---------------------------------------------------------------------------


def test_topk_encode_decode_shapes_and_finite():
    k = 4
    config = {"activation": "topk", "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "k": k}
    sae = GenericSAE(config, _simple_state_dict()).eval().to("cpu")
    x = _x()

    z = sae.encode(x)
    x_hat = sae.decode(z)

    assert z.shape == (BATCH, HIDDEN_DIM)
    assert x_hat.shape == (BATCH, INPUT_DIM)
    assert not torch.isnan(z).any()
    assert not torch.isnan(x_hat).any()

    # topk keeps at most k nonzero entries per row
    nnz_per_row = (z != 0).sum(dim=-1)
    assert int(nnz_per_row.max()) <= k


def test_batchtopk_encode_decode_shapes_and_finite():
    config = {
        "activation": "batchtopk",
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "threshold": 0.5,
    }
    sae = GenericSAE(config, _simple_state_dict()).eval().to("cpu")
    x = _x()

    z = sae.encode(x)
    x_hat = sae.decode(z)

    assert z.shape == (BATCH, HIDDEN_DIM)
    assert x_hat.shape == (BATCH, INPUT_DIM)
    assert not torch.isnan(z).any()
    assert not torch.isnan(x_hat).any()


def test_batchtopk_threshold_from_running_buffer():
    """threshold absent in config is read from the _running_threshold buffer."""
    sd = _simple_state_dict()
    sd["_running_threshold"] = torch.tensor(0.25)
    config = {"activation": "batchtopk", "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM}
    sae = GenericSAE(config, sd).eval().to("cpu")
    assert float(sae.threshold) == 0.25


def test_jumprelu_encode_decode_shapes_and_finite():
    sd = _simple_state_dict()
    sd["jump_thresholds"] = torch.full((HIDDEN_DIM,), 0.1)
    config = {"activation": "jumprelu", "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM}
    sae = GenericSAE(config, sd).eval().to("cpu")
    x = _x()

    z = sae.encode(x)
    x_hat = sae.decode(z)

    assert z.shape == (BATCH, HIDDEN_DIM)
    assert x_hat.shape == (BATCH, INPUT_DIM)
    assert not torch.isnan(z).any()
    assert not torch.isnan(x_hat).any()
    # thresholds come from the state dict, not the default
    assert torch.allclose(sae.jump_thresholds, torch.full((HIDDEN_DIM,), 0.1))


def test_jumprelu_defaults_when_absent():
    """With no thresholds in the state dict, jumprelu defaults to 0.01."""
    config = {"activation": "jumprelu", "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM}
    sae = GenericSAE(config, _simple_state_dict()).eval().to("cpu")
    assert torch.allclose(sae.jump_thresholds, torch.full((HIDDEN_DIM,), 0.01))


# ---------------------------------------------------------------------------
# Overcomplete format
# ---------------------------------------------------------------------------


def test_overcomplete_format_matches_simple_shapes():
    config = {"activation": "topk", "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "k": 4}
    sae = GenericSAE(config, _overcomplete_state_dict()).eval().to("cpu")
    x = _x()

    z = sae.encode(x)
    x_hat = sae.decode(z)

    assert z.shape == (BATCH, HIDDEN_DIM)
    assert x_hat.shape == (BATCH, INPUT_DIM)
    assert sae.decoder_bias is None  # overcomplete dictionary carries no bias
    assert not torch.isnan(x_hat).any()


# ---------------------------------------------------------------------------
# Dimension inference and Protocol conformance
# ---------------------------------------------------------------------------


def test_dimensions_inferred_from_weights():
    config = {"activation": "topk", "k": 4}  # no input_dim / hidden_dim
    sae = GenericSAE(config, _simple_state_dict())
    assert sae.input_dim == INPUT_DIM
    assert sae.hidden_dim == HIDDEN_DIM


def test_generic_sae_satisfies_protocol():
    config = {"activation": "topk", "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "k": 4}
    sae = GenericSAE(config, _simple_state_dict())
    assert isinstance(sae, SAEInterface)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def test_load_sae_from_weights_eval_mode():
    config = {"activation": "topk", "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "k": 4}
    sae = load_sae_from_weights(_simple_state_dict(), config, device="cpu")
    assert isinstance(sae, GenericSAE)
    assert sae.training is False
    x = _x()
    x_hat = sae.decode(sae.encode(x))
    assert x_hat.shape == (BATCH, INPUT_DIM)


def test_load_sae_local_dir_roundtrip(tmp_path):
    """load_sae from a local dir: sae.pt + config.json, no network."""
    sd = _simple_state_dict()
    torch.save(sd, tmp_path / "sae.pt")
    config = {"activation": "topk", "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "k": 4}
    (tmp_path / "config.json").write_text(json.dumps(config))

    sae = load_sae(str(tmp_path), device="cpu")
    assert isinstance(sae, GenericSAE)
    assert sae.training is False

    x = _x()
    z = sae.encode(x)
    x_hat = sae.decode(z)
    assert z.shape == (BATCH, HIDDEN_DIM)
    assert x_hat.shape == (BATCH, INPUT_DIM)
    assert not torch.isnan(x_hat).any()
