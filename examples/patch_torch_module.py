"""Replace the depthwise causal ``nn.Conv1d`` layers of a PyTorch model.

Models that do not depend on a fused kernel usually write a causal convolution as

    y = conv1d(x)[..., :seqlen]          # nn.Conv1d(dim, dim, width, groups=dim, padding=width - 1)

which computes width - 1 outputs too many and slices them off. ``patch`` finds those layers and
swaps in a module that calls the fused kernel on the same parameters. The replacement returns
seqlen outputs, so the slice that follows it in the model becomes a no-op and the model code does
not change.

    python examples/patch_torch_module.py
"""

import torch
from torch import nn
from causal_conv1d_cute import causal_conv1d_fn


class CausalConv1dCute(nn.Module):
    """Drop-in replacement for a depthwise causal nn.Conv1d that shares its parameters."""

    def __init__(self, conv, activation=None):
        """Keep a reference to the original layer so that its parameters stay registered."""
        super().__init__()
        self.conv, self.activation = (conv, activation)

    def forward(self, x):
        """Convolve a (batch, dim, seqlen) input."""
        return causal_conv1d_fn(x, self.conv.weight.squeeze(1), self.conv.bias, self.activation)


def is_depthwise_causal(module):
    """Return whether a module is an nn.Conv1d with one filter per channel and causal padding."""
    return (
        isinstance(module, nn.Conv1d)
        and module.groups == module.in_channels == module.out_channels
        and module.padding == (module.kernel_size[0] - 1,)
        and module.stride == (1,)
        and module.dilation == (1,)
        and module.padding_mode == "zeros"
    )


def patch(model):
    """Replace every depthwise causal nn.Conv1d in model and return how many were replaced."""
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if is_depthwise_causal(child):
                setattr(parent, name, CausalConv1dCute(child))
                count += 1
    return count


class Block(nn.Module):
    """A toy token mixer in the style of a short-convolution block."""

    def __init__(self, dim, width):
        """Create the projections and the depthwise convolution."""
        super().__init__()
        self.proj_in = nn.Linear(dim, dim, bias=False)
        self.conv = nn.Conv1d(dim, dim, width, groups=dim, padding=width - 1, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """Mix a (batch, seqlen, dim) input along the sequence."""
        seqlen = x.shape[1]
        h = self.proj_in(x).transpose(1, 2)
        h = nn.functional.silu(self.conv(h)[..., :seqlen])
        return self.proj_out(h.transpose(1, 2))


@torch.no_grad()
def main():
    """Patch a small model and check that its output is unchanged."""
    torch.manual_seed(0)
    model = nn.Sequential(*[Block(1024, 4) for _ in range(4)])
    for p in model.parameters():
        nn.init.normal_(p, std=0.7 if p.dim() == 3 else p.shape[-1] ** -0.5)
    model = model.cuda().to(torch.bfloat16)
    x = torch.randn(2, 2048, 1024, device="cuda", dtype=torch.bfloat16)
    before = model(x)
    print("replaced", patch(model), "layers")
    after = model(x)
    scale = before.float().abs().max().item()
    print(f"max |difference| = {(before.float() - after.float()).abs().max().item():.2e}")
    print(f"max |output|     = {scale:.2e}")
    assert (before.float() - after.float()).abs().max().item() <= 0.02 * scale


if __name__ == "__main__":
    main()
