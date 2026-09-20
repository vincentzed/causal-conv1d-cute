"""Patch code written against the Dao-AILab causal_conv1d interface.

mamba_ssm, flash-linear-attention and the fast paths of Hugging Face transformers all import
``causal_conv1d_fn`` and ``causal_conv1d_update`` from the ``causal_conv1d`` package. Calling
``dao_compat.install()`` after those imports rebinds the two functions in every module that holds
them. Calls the kernels cannot serve, such as training steps or calls with seq_idx, still reach
the upstream implementation.

The layer below stands in for such model code. The script runs it before and after patching and
checks that prefill outputs, decode outputs and conv states agree.

    pip install causal-conv1d causal-conv1d-cute
    python examples/patch_upstream_api.py
"""

import torch
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from causal_conv1d_cute import dao_compat


class ShortConv(torch.nn.Module):
    """A depthwise causal convolution layer written against the upstream interface."""

    def __init__(self, dim, width):
        """Create the filter weights."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim, width) * 0.3)

    def prefill(self, x):
        """Convolve a (batch, dim, seqlen) input and return the output and the final conv state."""
        width = self.weight.shape[1]
        return causal_conv1d_fn(x, self.weight, activation="silu"), x[..., 1 - width :].clone()

    def decode(self, x, conv_state):
        """Advance conv_state by one (batch, dim) token and return its output."""
        return causal_conv1d_update(x, conv_state, self.weight, activation="silu")


@torch.no_grad()
def run(layer, x, token):
    """Prefill a batch, then decode one token."""
    y, state = layer.prefill(x)
    return y, layer.decode(token, state), state


def main():
    """Compare the layer before and after patching."""
    torch.manual_seed(0)
    layer = ShortConv(2048, 4).cuda().to(torch.bfloat16)
    x = torch.randn(2, 2048, 512, device="cuda", dtype=torch.bfloat16)
    token = torch.randn(2, 2048, device="cuda", dtype=torch.bfloat16)
    before = run(layer, x, token)
    print("rebound", dao_compat.install(), "references")
    after = run(layer, x, token)
    print("calls served by the kernels:", dao_compat.stats)
    for name, a, b in zip(("prefill", "decode", "conv state"), before, after):
        print(f"{name:<11} max |difference| = {(a.float() - b.float()).abs().max().item():.4f}")
    assert torch.equal(before[2], after[2])
    assert dao_compat.stats["fn_fast"] == 1 and dao_compat.stats["update_fast"] == 1


if __name__ == "__main__":
    main()
