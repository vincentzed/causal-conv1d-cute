"""Provide depthwise causal 1D convolution operators.

Exports the CausalConv1d module and functional interfaces: causal_conv1d_fn: Convolution on dense
sequences. causal_conv1d_update: State update for decoding, one or several tokens.
causal_conv1d_update_ring: Single-token decode on a ring conv state, with to_ring and from_ring to
convert states. causal_conv1d_varlen_fn: Convolution on variable-length sequences. supported:
Kernel support query function.
"""

__version__ = "0.1.0"
from .api import CausalConv1d
from .functional import (
    causal_conv1d_fn,
    causal_conv1d_update,
    causal_conv1d_update_ring,
    causal_conv1d_varlen_fn,
    supported,
)
from .ring import from_ring, to_ring

__all__ = [
    "CausalConv1d",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "causal_conv1d_update_ring",
    "causal_conv1d_varlen_fn",
    "from_ring",
    "supported",
    "to_ring",
]
