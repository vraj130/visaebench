# Custom models

VISAEBench is deliberately unopinionated about where your SAE and your ViT
come from. This page covers three things: plugging in your own SAE via the
`SAEInterface` Protocol, the two checkpoint formats `GenericSAE` accepts,
and wrapping an arbitrary ViT with `CustomBackbone`.

## Your own SAE via the Protocol

`SAEInterface` (`visaebench.core.types`) is a `@runtime_checkable`
`Protocol`. Any object that implements `encode` and `decode` with the
right shapes satisfies it. No inheritance, no registration, no base class.

- `encode(x)`: takes activations `[batch, d_model]` and returns sparse
  latent codes `[batch, d_sae]`.
- `decode(z)`: takes codes `[batch, d_sae]` and returns reconstructed
  activations `[batch, d_model]`.

A plain `nn.Module` with those two methods passes
`isinstance(model, visaebench.SAEInterface)`:

```python
import torch
import visaebench


class MySAE(torch.nn.Module):
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.encoder = torch.nn.Linear(d_model, d_sae)
        self.decoder = torch.nn.Linear(d_sae, d_model)

    def encode(self, x):
        pre = torch.relu(self.encoder(x))
        vals, idx = pre.topk(self.k, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, idx, vals)
        return sparse

    def decode(self, z):
        return self.decoder(z)


sae = MySAE(d_model=768, d_sae=768 * 8, k=32)
assert isinstance(sae, visaebench.SAEInterface)  # passes, no inheritance
```

Once it satisfies the Protocol, pass it straight to `evaluate`. See
`examples/02_evaluate_your_own_sae.py` for a complete CPU-only run that
scores such an SAE on FVU without any downloads.

## Checkpoint formats accepted by GenericSAE

If your weights live on disk or the Hub rather than in a live object, use
`visaebench.hub.load_sae`, which builds a `GenericSAE`. `GenericSAE`
implements the `SAEInterface` and supports three inference activations
(`topk`, `batchtopk`, `jumprelu`) plus two state-dict key conventions.
Both conventions are handled in `_extract_weights`.

Local checkpoints often ship `config.yaml` instead of `config.json`. Pass
`config_filename="config.yaml"` in that case; the loader branches on the
file suffix and reads either JSON or YAML.

### Simple format (the VISAEBench HF upload format)

```
encoder.weight   [hidden_dim, input_dim]
encoder.bias     [hidden_dim]
decoder.weight   [hidden_dim, input_dim]
decoder.bias     [input_dim]           (optional)
```

### Overcomplete library format

```
encoder.final_block.0.weight   [hidden_dim, input_dim]
encoder.final_block.0.bias     [hidden_dim]
dictionary._weights            [hidden_dim, input_dim]
dictionary.multiplier          scalar   (optional)
```

For the overcomplete format the dictionary has no decoder bias. When
`dictionary.multiplier` is present and non-zero, the decoder weight is
scaled by `multiplier.exp()` before use.

### Config and activation

`input_dim` and `hidden_dim` are inferred from the encoder weight when not
present in the config. The activation is taken from `config["activation"]`
(falling back to `sae_type` or `architecture`), lowercased. For BatchTopK
the threshold is read from `config["threshold"]` if present, otherwise
from a `_running_threshold` buffer in the state dict, otherwise defaulted
to 0.0. For JumpReLU per-concept thresholds are read from
`jump_thresholds` or `thresholds` in the state dict.

Loading, from the Hub or a local directory:

```python
from visaebench.hub import load_sae

# From the Hub, one repo per backbone with per-config subfolders.
sae = load_sae("visaebench/clip-vitb16-saes", subfolder="batchtopk_16x_k128")

# From a local checkpoint that ships config.yaml.
sae = load_sae("/data/checkpoints/my_sae/", config_filename="config.yaml")
```

If you already hold weights in memory, `load_sae_from_weights(state_dict,
config)` builds a `GenericSAE` directly.

## Wrapping arbitrary ViTs with CustomBackbone

`CustomBackbone` (`visaebench.backbones.custom`) adapts any HuggingFace or
timm ViT into a backbone that `evaluate` can extract from. It does not
auto-register, so you instantiate it directly.

```python
from visaebench.backbones import CustomBackbone

# HuggingFace
hf_backbone = CustomBackbone(
    "google/vit-base-patch16-224",
    source="huggingface",
    device="cuda",
)

# timm
timm_backbone = CustomBackbone(
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    source="timm",
    device="cuda",
)
```

Constructor signature:

```python
CustomBackbone(
    model_name,
    *,
    source="huggingface",   # "huggingface" or "timm"
    num_prefix_tokens=1,    # non-spatial prefix tokens to drop
    device="cuda",
)
```

Pass the instance to `evaluate` through `backbone=`:

```python
import visaebench

results = visaebench.evaluate(
    sae=sae,
    backbone=hf_backbone,
    backbone_name="clip_vitb16",  # still required, see below
    imagenet_path=None,
    device="cuda",
)
```

`backbone_name` is still required even when you pass a custom instance. It
is not used for extraction (your instance is), but it is the activation
cache key and it selects the M5 cross-model evaluator through the internal
cross-model map. Choose the registered name whose cross-model pairing you
want.

### The num_prefix_tokens gotcha

`extract_activations` returns patch tokens only, so it strips leading
non-spatial prefix tokens from the hidden states before returning them.
`num_prefix_tokens` controls how many are dropped, and it defaults to 1
(a single CLS token), which is correct for a plain ViT-B/16.

If your model has more than one prefix token, you must set this
explicitly, or spatial tokens will be misaligned (the leftover prefix
token would be treated as a patch, shifting the whole grid and breaking
the square `grid_size` inference). Common cases:

- DeiT with a distillation token: the model prepends both a CLS token and
  a distillation token, so pass `num_prefix_tokens=2`.
- DINOv2 with register tokens: pass `1 + num_register_tokens` (for
  example `num_prefix_tokens=5` for four registers plus CLS).

When `num_prefix_tokens` is 0, nothing is stripped and all tokens are
returned as patch tokens.
