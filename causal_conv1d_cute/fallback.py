"""Compute causal 1D convolutions with PyTorch operators.

The functional interface calls these when the CuTe DSL kernels cannot serve a call, for example on
a channel count that is not a multiple of 16, on a CPU tensor, or on a kernel width above 8. They
follow the same conventions as the kernels: sums are accumulated in float32 and cast back to the
input dtype, conv states are updated in place, and entries whose slot equals pad_slot_id are
skipped. They loop over sequences on the host and are not meant to be fast.
"""

import warnings
import torch
import torch.nn.functional as F

_warned = set()


def warn(why):
    """Warn once per distinct reason that a call is served by PyTorch operators.

    Args:
        why: Description of the requirement that the call does not meet.
    """
    if why not in _warned:
        _warned.add(why)
        warnings.warn(f"causal_conv1d_cute: using the PyTorch fallback ({why})", stacklevel=3)


def _conv(full, weight, bias, activation):
    """Convolve one (dim, tokens) sequence without padding, accumulating in float32.

    Args:
        full: Input of shape (dim, tokens), already preceded by width - 1 elements of history.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: "silu" or None.

    Returns:
        Float32 output of shape (dim, tokens - width + 1).
    """
    y = F.conv1d(
        full.float().unsqueeze(0),
        weight.float().unsqueeze(1),
        None if bias is None else bias.float(),
        groups=weight.shape[0],
    )[0]
    return F.silu(y) if activation else y


def fwd(x, weight, bias, activation, out=None):
    """Compute causal 1D convolution over batched sequences.

    Args:
        x: Input tensor of shape (batch, dim, seqlen).
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: "silu" or None.
        out: Optional output tensor of shape (batch, dim, seqlen).

    Returns:
        Output tensor of shape (batch, dim, seqlen).
    """
    D, W = weight.shape
    y = F.conv1d(
        x.float(),
        weight.float().unsqueeze(1),
        None if bias is None else bias.float(),
        padding=W - 1,
        groups=D,
    )[..., : x.shape[-1]]
    y = (F.silu(y) if activation else y).to(x.dtype)
    if out is None:
        return y
    out.copy_(y)
    return out


def varlen(
    x, weight, bias, query_start_loc, cache_indices, has_initial_state, conv_states, activation, pad
):
    """Compute causal 1D convolution over a variable-length packed batch.

    Args:
        x: Packed input tensor of shape (dim, total_seqlen).
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        query_start_loc: Cumulative sequence lengths of shape (batch + 1,).
        cache_indices: Optional tensor of shape (batch,) mapping sequences to conv state slots.
        has_initial_state: Optional tensor of shape (batch,) marking sequences that read their
            initial state from conv_states.
        conv_states: Optional conv state tensor of shape (slots, dim, width - 1), updated in place.
        activation: "silu" or None.
        pad: Slot index marking sequences to skip.

    Returns:
        Output tensor of shape (dim, total_seqlen).
    """
    D, W = weight.shape
    K = W - 1
    out = torch.empty_like(x)
    bounds = query_start_loc.tolist()
    nseq = len(bounds) - 1
    slots = list(range(nseq)) if cache_indices is None else cache_indices.tolist()
    inits = [False] * nseq if has_initial_state is None else has_initial_state.tolist()
    for i in range(nseq):
        a, e, slot = (bounds[i], bounds[i + 1], slots[i])
        if e <= a or (conv_states is not None and slot == pad):
            continue
        use = conv_states is not None and bool(inits[i])
        init = conv_states[slot] if use else x.new_zeros(D, K)
        full = torch.cat([init, x[:, a:e]], dim=1)
        out[:, a:e] = _conv(full, weight, bias, activation).to(x.dtype)
        if conv_states is not None:
            conv_states[slot] = full[:, -K:]
    return out


def update(
    x,
    conv_state,
    weight,
    bias,
    activation,
    conv_state_indices,
    pad,
    intermediate_conv_window,
    intermediate_state_indices,
):
    """Advance the conv state by one or more tokens and compute their outputs.

    Args:
        x: Input tensor of shape (batch, dim) or (batch, dim, steps).
        conv_state: Conv state tensor of shape (slots, dim, state_len) with state_len of at least
            width - 1, updated in place.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: "silu" or None.
        conv_state_indices: Optional tensor of shape (batch,) mapping batch rows to slots.
        pad: Slot index marking batch rows to skip.
        intermediate_conv_window: Optional cache of shape (slots, steps, dim, width - 1) that
            receives the conv state after every token.
        intermediate_state_indices: Optional tensor of shape (batch,) mapping batch rows to slots
            of intermediate_conv_window. Defaults to conv_state_indices.

    Returns:
        Output tensor with the shape of x.
    """
    W = weight.shape[1]
    K = W - 1
    xs = x.unsqueeze(-1) if x.dim() == 2 else x
    B, _, L = xs.shape
    out = torch.empty_like(xs)
    slots = list(range(B)) if conv_state_indices is None else conv_state_indices.tolist()
    islots = slots if intermediate_state_indices is None else intermediate_state_indices.tolist()
    for i, slot in enumerate(slots):
        if slot == pad:
            continue
        full = torch.cat([conv_state[slot], xs[i]], dim=1)
        out[i] = _conv(full[:, -(K + L) :], weight, bias, activation).to(x.dtype)
        conv_state[slot] = full[:, L:]
        if intermediate_conv_window is not None:
            for t in range(L):
                first = full.shape[1] - L + t + 1 - K
                intermediate_conv_window[islots[i], t] = full[:, first : first + K]
    return out.squeeze(-1) if x.dim() == 2 else out
