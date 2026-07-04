"""Evaluate your own SAE on CPU with zero downloads (FVU only).

What this does
    Shows the smallest possible integration: a custom SAE class that
    implements ONLY ``encode`` and ``decode``, wrapped around random
    weights, scored with the reconstruction metric (M2, FVU) on a random
    activation tensor.  No ImageNet, no network, no GPU.

Why it is useful
    ``SAEInterface`` is a structural (duck-typed) Protocol: any object with
    ``encode(x)`` and ``decode(z)`` satisfies it, with no base class to
    inherit from.  This example demonstrates that directly and gives you a
    template for plugging in your own architecture.

Hardware
    CPU only.  No GPU required.

Runtime
    A few seconds (small random tensors, one metric).

Run with
    PYTHONPATH=/mnt/NAS/home/vg2097/visaebench \\
        python examples/02_evaluate_your_own_sae.py
"""

import torch

import visaebench


# ----------------------------------------------------------------------
# 1. Define a minimal SAE.
#
# Note: NO inheritance.  We do not subclass anything from visaebench.
# The only requirement is that the object exposes ``encode`` and
# ``decode`` with the right shapes, which is exactly what the
# SAEInterface Protocol checks for structurally.
# ----------------------------------------------------------------------
class MyTinySAE(torch.nn.Module):
    """A top-k SAE with random weights, for demonstration only.

    encode: [batch, input_dim] -> [batch, hidden_dim] (sparse)
    decode: [batch, hidden_dim] -> [batch, input_dim]
    """

    def __init__(self, input_dim: int, hidden_dim: int, k: int) -> None:
        super().__init__()
        self.k = k
        # Encoder maps model activations up into the (overcomplete) dictionary.
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        # Decoder maps sparse codes back down to model-activation space.
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Linear projection, ReLU, then keep only the top-k values per row.
        pre = torch.relu(self.encoder(x))
        vals, idx = pre.topk(self.k, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, idx, vals)
        return sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ----------------------------------------------------------------------
# 2. Instantiate the SAE.
#
# input_dim = 768 matches ViT-B hidden size; hidden_dim is an 8x
# expansion, a common SAE dictionary width.
# ----------------------------------------------------------------------
INPUT_DIM = 768
HIDDEN_DIM = 768 * 8
sae = MyTinySAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, k=32)
sae.eval()

# Confirm the structural Protocol check passes without any inheritance.
assert isinstance(sae, visaebench.SAEInterface)


# ----------------------------------------------------------------------
# 3. Build a random 2-D activation tensor [N, D].
#
# FVU accepts a flat [N, D] token matrix directly, so we skip images,
# labels, and the backbone entirely.  With mean=None / std=None (the
# defaults) FVU treats these activations as already normalised.
# ----------------------------------------------------------------------
torch.manual_seed(0)
acts = torch.randn(4000, INPUT_DIM)


# ----------------------------------------------------------------------
# 4. Run the benchmark, restricted to FVU on CPU.
#
# Passing ``activations`` bypasses ImageNet extraction; ``ood_datasets=[]``
# skips all OOD downloads; ``metrics=["fvu"]`` avoids metrics that need
# images or labels; ``grid_size=(1, 1)`` is a harmless placeholder since
# FVU does not use the patch grid.
# ----------------------------------------------------------------------
results = visaebench.evaluate(
    sae=sae,
    activations=acts,
    labels=None,
    metrics=["fvu"],
    device="cpu",
    ood_datasets=[],
    grid_size=(1, 1),
)

# ``evaluate`` already prints the summary, but we print it again here so
# the value is obvious when reading the script's output.
print(results.summary())
