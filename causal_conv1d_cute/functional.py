"""Provide functional interfaces for causal 1D convolutions.

Packed filter weights are initialized once per (weight, bias) pair and cached across invocations.
"""

import functools
import math
import torch
from . import fallback as FB
from . import fwd as F
from . import ring as R
from . import varlen as V
from .api import CausalConv1d

PAD_SLOT_ID = -1
MAX_STEPS = 16
_layers = {}
_MAX_LAYERS = 1024


def _act(activation):
    """Normalize activation argument to canonical name.

    Args:
        activation: Activation name ("silu", "swish"), bool, or None.

    Returns:
        Canonical activation string ("silu") or None.

    Raises:
        ValueError: If activation is not one of the accepted values.
    """
    if activation in ("silu", "swish") or activation is True:
        return "silu"
    if activation in (None, False):
        return None
    raise ValueError(f'activation must be None, "silu" or "swish", got {activation!r}')


def _layer(weight, bias, activation):
    """Return the cached CausalConv1d for (weight, bias, activation), building it on first use.

    Callers usually pass a fresh view of the parameter on every forward (for example
    conv.weight.view(dim, width)), so the cache is keyed on the data pointer, shape, dtype and
    version counter instead of object identity. Each entry keeps a reference to the tensors it was
    built from, which pins their storage: the pointer in the key cannot be handed to unrelated
    memory while the entry is alive. In-place updates bump the version counter and miss the cache.
    """
    key = (
        weight.data_ptr(),
        tuple(weight.shape),
        weight.dtype,
        weight._version,
        None if bias is None else (bias.data_ptr(), bias._version),
        activation,
    )
    hit = _layers.get(key)
    if hit is None:
        if len(_layers) >= _MAX_LAYERS:
            _layers.pop(next(iter(_layers)))
        hit = _layers[key] = (weight, bias, CausalConv1d(weight, bias, activation))
    return hit[2]


def _check(x, weight, bias, channel_dim):
    """Reject arguments whose shapes are inconsistent with a depthwise causal convolution.

    Args:
        x: Input tensor.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        channel_dim: Dimension of x that indexes channels.

    Raises:
        ValueError: If weight is not two-dimensional, the kernel width is below 2, x does not have
            dim channels along channel_dim, or bias does not have shape (dim,).
    """
    if weight.dim() != 2 or weight.shape[1] < 2:
        raise ValueError(f"weight must have shape (dim, width >= 2), got {tuple(weight.shape)}")
    D = weight.shape[0]
    if x.dim() <= channel_dim or x.shape[channel_dim] != D:
        raise ValueError(f"input shape {tuple(x.shape)} does not have {D} channels")
    if bias is not None and tuple(bias.shape) != (D,):
        raise ValueError(f"bias must have shape ({D},), got {tuple(bias.shape)}")


def _reason(x, weight, bias=None):
    """Return why the tensors cannot be served by the kernels, or None when they can.

    Args:
        x: Input tensor.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).

    Returns:
        A short description of the first unmet requirement, or None.
    """
    D, W = weight.shape
    if torch.is_grad_enabled() and (
        x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    ):
        return "gradients are required and the kernels have no backward pass"
    if not x.is_cuda:
        return "input is not a CUDA tensor"
    if x.dtype not in F._DTYPES:
        return f"dtype {x.dtype} is not bf16, fp16 or fp32"
    if weight.dtype != x.dtype or not weight.is_contiguous():
        return "weight must be contiguous and share the input dtype"
    if bias is not None and (bias.dtype != x.dtype or not bias.is_contiguous()):
        return "bias must be contiguous and share the input dtype"
    if D % 16 != 0:
        return f"{D} channels is not a multiple of 16"
    if not 2 <= W <= 8:
        return f"kernel width {W} is outside 2..8"
    return None


def supported(x, weight, bias=None):
    """Check if input tensors satisfy causal 1D convolution requirements.

    Args:
        x: Input tensor on a CUDA device.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).

    Returns:
        True if x is on CUDA with supported dtype, filter weights match dtype
        and are contiguous, bias matches dtype and is contiguous when present,
        channels dimension is a multiple of 16, and kernel width is between 2
        and 8 inclusive; False otherwise.
    """
    return _reason(x, weight, bias) is None


def _state_reason(conv_state, D, W, dtype):
    """Return why a conv state cache cannot be served, or None when it can.

    Args:
        conv_state: Conv state tensor.
        D: Number of channels.
        W: Kernel width.
        dtype: Expected data type.

    Returns:
        A short description of the first unmet requirement, or None.
    """
    if conv_state.dim() != 3 or tuple(conv_state.shape[1:]) != (D, W - 1):
        return f"conv state shape {tuple(conv_state.shape)} is not (slots, {D}, {W - 1})"
    if conv_state.dtype != dtype:
        return "conv state dtype differs from the input dtype"
    if conv_state.stride(2) != 1 or conv_state.stride(1) != W - 1:
        return f"conv state strides {conv_state.stride()} are not channel-major"
    return None


def _state_ok(conv_state, D, W, dtype):
    """Check whether conv state matches expected layout and dtype.

    Args:
        conv_state: Conv state tensor.
        D: Number of channels.
        W: Kernel width.
        dtype: Expected data type.

    Returns:
        True if conv_state has 3 dimensions, shape (*, D, W - 1), matching dtype,
        and contiguous innermost strides; False otherwise.
    """
    return _state_reason(conv_state, D, W, dtype) is None


def _tiles_ok(t, vec=16):
    """Check that the rows of a two-dimensional tensor can be addressed as aligned tiles.

    Args:
        t: Tensor whose last dimension is divided into tiles.
        vec: Number of elements per tile.

    Returns:
        True if the last dimension has unit stride, the row stride is a multiple of vec, and the
        first element is aligned to one tile.
    """
    return (
        t.stride(-1) == 1
        and (t.dim() < 2 or t.stride(-2) % vec == 0)
        and t.data_ptr() % (vec * t.element_size()) == 0
    )


def varlen_reason(x, weight, bias, query_start_loc, conv_states):
    """Return why a variable-length call cannot be served, or None when it can.

    Args:
        x: Input tensor of shape (dim, total_seqlen).
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        query_start_loc: Sequence start offsets tensor of shape (batch + 1,).
        conv_states: Optional conv state tensor of shape (slots, dim, width - 1).

    Returns:
        A short description of the first unmet requirement, or None.
    """
    D, W = weight.shape
    if x.dim() != 2 or x.shape[0] != D:
        return f"input shape {tuple(x.shape)} is not ({D}, total_seqlen)"
    if query_start_loc is None or query_start_loc.dtype != torch.int32:
        return "query_start_loc must be an int32 tensor"
    if x.shape[1] < 16:
        return f"{x.shape[1]} tokens is fewer than 16"
    why = _reason(x, weight, bias)
    if why is None and conv_states is not None:
        why = _state_reason(conv_states, D, W, x.dtype)
    return why


def varlen_supported(x, weight, bias, query_start_loc, conv_states):
    """Check if inputs satisfy variable-length convolution requirements.

    Args:
        x: Input tensor of shape (dim, total_seqlen).
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        query_start_loc: Sequence start offsets tensor of shape (batch + 1,).
        conv_states: Optional conv state tensor of shape (slots, dim, width - 1).

    Returns:
        True if x is 2D with first dimension equal to channels, query_start_loc
        is int32, sequence length is at least 16, supported() returns True, and
        conv_states meets shape and stride requirements; False otherwise.
    """
    return varlen_reason(x, weight, bias, query_start_loc, conv_states) is None


def update_reason(x, conv_state, weight, bias, intermediate_conv_window=None):
    """Return why a decode call cannot be served, or None when it can.

    Args:
        x: Input tensor of shape (batch, dim), or (batch, dim, steps) viewing contiguous (batch,
            steps, dim) storage.
        conv_state: Conv state tensor of shape (slots, dim, width - 1).
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        intermediate_conv_window: Optional per-token state cache of shape (slots, steps, dim,
            width - 1).

    Returns:
        A short description of the first unmet requirement, or None.
    """
    D, W = weight.shape
    why = _reason(x, weight, bias) or _state_reason(conv_state, D, W, x.dtype)
    if why is not None:
        return why
    if x.dim() == 2:
        if x.shape[1] != D or x.stride(1) != 1:
            return f"input shape {tuple(x.shape)} strides {x.stride()} is not (batch, {D}) rows"
        if intermediate_conv_window is not None:
            return "per-token states need a (batch, dim, steps) input"
        return None
    if x.dim() != 3 or x.shape[1] != D:
        return f"input shape {tuple(x.shape)} is not (batch, {D}, steps)"
    L = x.shape[2]
    if not 1 <= L <= MAX_STEPS:
        return f"{L} steps is outside 1..{MAX_STEPS}"
    if not x.transpose(1, 2).is_contiguous() or not _tiles_ok(x.transpose(1, 2)):
        return f"input strides {x.stride()} do not view aligned (batch, steps, dim) storage"
    if not _tiles_ok(conv_state.view(conv_state.shape[0], -1)):
        return "conv state is not aligned to 16 elements"
    inter = intermediate_conv_window
    if inter is not None and _inter_layout(inter, D, W, L, x.dtype) is None:
        return f"per-token state shape {tuple(inter.shape)} strides {inter.stride()} unsupported"
    return None


def update_supported(x, conv_state, weight, bias):
    """Check if inputs satisfy state update kernel requirements.

    Args:
        x: Input tensor of shape (batch, dim) or (batch, dim, steps).
        conv_state: Conv state tensor of shape (slots, dim, width - 1).
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).

    Returns:
        True if update_reason() finds no unmet requirement; False otherwise.
    """
    return update_reason(x, conv_state, weight, bias) is None


def _inter_layout(inter, D, W, L, dtype):
    """Classify the layout of a per-token state cache.

    Args:
        inter: Per-token state cache of shape (slots, steps, dim, width - 1).
        D: Number of channels.
        W: Kernel width.
        L: Number of tokens per sequence.
        dtype: Expected data type.

    Returns:
        "dense" for contiguous storage, "dedup" when consecutive states overlap in rows of
        steps + width - 2 elements per channel, or None when the layout is not supported.
    """
    if inter.dim() != 4 or tuple(inter.shape[1:]) != (L, D, W - 1) or inter.dtype != dtype:
        return None
    if inter.data_ptr() % (16 * inter.element_size()) != 0:
        return None
    PW = L + W - 2
    if inter.stride() == (D * PW, 1, PW, 1):
        return "dedup"
    if inter.is_contiguous():
        return "dense"
    return None


def causal_conv1d_fn(x, weight, bias=None, activation=None, out=None):
    """Compute causal 1D convolution over batched sequences.

    Args:
        x: Input tensor of shape (batch, dim, seqlen) contiguous, or a (batch, dim, seqlen) view of
            contiguous (batch, seqlen, dim) storage.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: Optional activation: "silu", "swish", or None.
        out: Optional output tensor of shape (batch, dim, seqlen).

    Returns:
        Output tensor of shape (batch, dim, seqlen).

    Raises:
        ValueError: If the shapes of x, weight and bias are inconsistent, or out shares storage
            with x.
    """
    _check(x, weight, bias, 1)
    if x.dim() != 3:
        raise ValueError(f"input must have shape (batch, dim, seqlen), got {tuple(x.shape)}")
    act = _act(activation)
    why = _reason(x, weight, bias)
    if why is not None:
        FB.warn(why)
        return FB.fwd(x, weight, bias, act, out)
    if not (
        x.is_contiguous()
        or x.transpose(1, 2).is_contiguous()
        or F.channellast_row_stride(x) is not None
    ):
        x = x.contiguous()
    return _layer(weight, bias, act).fwd(x, out=out)


def causal_conv1d_varlen_fn(
    x,
    weight,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    conv_states=None,
    activation=None,
    pad_slot_id=PAD_SLOT_ID,
):
    """Compute causal 1D convolution over a variable-length packed batch.

    The packed input is convolved as a single sequence using the forward kernel, followed by a
    correction pass across sequence boundaries that reads and updates per-sequence conv states in
    place.

    A (dim, total_seqlen) view of token-major storage is consumed without a copy, including the
    leading dim columns of a wider (total_seqlen, cols) buffer; the output then views contiguous
    (total_seqlen, dim) storage. Any other non-contiguous input is copied first.

    Args:
        x: Packed input tensor of shape (dim, total_seqlen).
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        query_start_loc: Cumulative sequence lengths of shape (batch + 1,), dtype torch.int32.
        cache_indices: Optional int32 tensor of shape (batch,) mapping sequence indices to conv
            state slot indices.
        has_initial_state: Optional bool or uint8 tensor of shape (batch,) indicating whether each
            sequence reads an initial state from conv_states.
        conv_states: Optional conv state tensor of shape (slots, dim, width - 1), updated in place.
        activation: Optional activation: "silu", "swish", or None.
        pad_slot_id: Slot index representing padded entries to skip during conv state updates.

    Returns:
        Output tensor of shape (dim, total_seqlen).

    Raises:
        ValueError: If the shapes of x, weight and bias are inconsistent or query_start_loc is
            missing.
    """
    _check(x, weight, bias, 0)
    if x.dim() != 2 or query_start_loc is None:
        raise ValueError("a packed batch needs x of shape (dim, total_seqlen) and query_start_loc")
    why = varlen_reason(x, weight, bias, query_start_loc, conv_states)
    if why is not None:
        FB.warn(why)
        return FB.varlen(
            x,
            weight,
            bias,
            query_start_loc,
            cache_indices,
            has_initial_state,
            conv_states,
            _act(activation),
            pad_slot_id,
        )
    return _varlen(
        x,
        weight,
        bias,
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_states,
        activation,
        pad_slot_id,
    )


def _varlen(
    x,
    weight,
    bias,
    query_start_loc,
    cache_indices,
    has_initial_state,
    conv_states,
    activation,
    pad_slot_id,
):
    """Run causal_conv1d_varlen_fn on arguments that varlen_reason() has already accepted."""
    D, W = weight.shape
    act = _act(activation)
    layer = _layer(weight, bias, act)
    T = x.shape[1]
    channel_last = (
        x.stride(0) == 1
        and x.stride(1) >= D
        and x.stride(1) % 4 == 0
        and x.data_ptr() % (16 * x.element_size()) == 0
    )
    nseq = query_start_loc.numel() - 1
    has_state = conv_states is not None
    if channel_last:
        aligned = has_state and conv_states.data_ptr() % (16 * x.element_size()) == 0
        if aligned:
            return _varlen_token_major(
                layer,
                x,
                query_start_loc,
                cache_indices,
                has_initial_state,
                conv_states,
                pad_slot_id,
                nseq,
            )
        out = torch.empty((T, D), dtype=x.dtype, device=x.device).t()
        strip, bs, macro = _seq_config(T, W)
        layer.fwd(x.unsqueeze(0), out=out.unsqueeze(0), cfg=("strip", strip, bs, macro))
    else:
        if x.stride(-1) != 1:
            x = x.contiguous()
        out = torch.empty_like(x)
        layer.fwd(x.unsqueeze(0), out=out.unsqueeze(0))
    if has_state:
        if cache_indices is None:
            cache_indices = torch.arange(nseq, device=x.device, dtype=torch.int32)
        elif cache_indices.dtype != torch.int32:
            cache_indices = cache_indices.to(torch.int32)
        hinit = (
            torch.zeros(nseq, device=x.device, dtype=torch.uint8)
            if has_initial_state is None
            else has_initial_state.view(torch.uint8)
            if has_initial_state.dtype == torch.bool
            else has_initial_state.to(torch.uint8)
        )
        state = conv_states
    else:
        cache_indices, hinit = (query_start_loc, query_start_loc.view(torch.uint8)[:nseq])
        state = x.new_empty(1, D, W - 1)
    k = V._compile_boundary(
        x.dtype,
        weight.dtype,
        D,
        W,
        bias is not None,
        act is not None,
        has_state,
        channel_last=channel_last,
    )
    k(x, out, layer.wp, query_start_loc, cache_indices, hinit, state, T, int(pad_slot_id), nseq)
    return out


def _varlen_token_major(
    layer, x, query_start_loc, cache_indices, has_initial_state, conv_states, pad_slot_id, nseq
):
    """Run a packed batch whose input views token-major storage and whose states are cached.

    One sequence is served by a single kernel launch. Several sequences take two: the strip walk
    over the packed stream, then the tiled boundary correction. Everything that depends only on
    the token count, the row spacing of x and the number of launches is resolved once per layer
    and kept in a plan, so that a call costs one allocation, the views and the launches.

    Args:
        layer: The CausalConv1d holding the packed weights.
        x: Packed input of shape (dim, total_seqlen) with adjacent channels one element apart.
        query_start_loc: Cumulative sequence lengths of shape (batch + 1,), dtype torch.int32.
        cache_indices: Optional tensor of shape (batch,) mapping sequences to conv state slots.
        has_initial_state: Optional tensor of shape (batch,) marking sequences that start from
            their cached state.
        conv_states: Conv state tensor of shape (slots, dim, width - 1), aligned to 16 elements.
        pad_slot_id: Slot index marking sequences to skip.
        nseq: Number of sequences.

    Returns:
        Output of shape (dim, total_seqlen) viewing contiguous (total_seqlen, dim) storage.
    """
    D, W = (layer.D, layer.W)
    T, xs = (x.shape[1], x.stride(1))
    single = nseq == 1
    key = (T, xs, single, x.dtype)
    plan = layer._vplans.get(key)
    wflat = layer.wflat
    weights = layer.wtm if wflat is None else wflat
    if plan is None:
        strip, bs, macro = _seq_config(T, W)
        span = strip * macro
        nstrips = (T + span - 1) // span
        vec = math.gcd(16, xs)
        threads = nstrips * (D // vec)
        walks, fix = _token_major_kernels(
            x.dtype,
            D,
            W,
            layer.has_bias,
            layer.act is not None,
            None if xs == D else xs,
            "time" if wflat is None else "channel",
            vec,
        )
        walk = walks[single, macro]
        plan = (walk, fix, (T - 1) * xs + D, nstrips, (threads + bs - 1) // bs)
        if len(layer._vplans) < 4096:
            layer._vplans[key] = plan
    walk, fix, n, nstrips, nblocks = plan
    buf = torch.empty((T, D), dtype=x.dtype, device=x.device)
    xflat = x.as_strided((1, n), (n, 1))
    oflat = buf.view(1, -1)
    state = conv_states.view(conv_states.shape[0], -1)
    cidx = _int32(cache_indices, x.device, nseq)
    hinit = _flags(has_initial_state, x.device, nseq)
    pad = int(pad_slot_id)
    if single:
        walk(
            xflat,
            weights,
            weights,
            oflat,
            query_start_loc,
            state,
            cidx,
            hinit,
            T,
            nstrips,
            nblocks,
            pad,
        )
    else:
        walk(xflat, weights, oflat, T, nstrips, nblocks, 1)
        fix(xflat, oflat, weights, query_start_loc, cidx, hinit, state, T, pad, nseq)
    return buf.t()


_MACROS = (1, 2, 4, 8, 6)


@functools.cache
def _token_major_kernels(dtype, D, W, has_bias, silu, xstride, wmajor="time", vec=16):
    """Compile every kernel that a token-major packed batch of one layer shape can need.

    A serving engine calls the prefill path with token counts that are not known in advance. The
    kernels depend on the token count only through the macro count, so all of them are compiled
    on the first call, which an engine issues while it starts up, instead of one at a time in the
    middle of serving, where each compilation would stall every request in flight.

    Args:
        dtype: Activation and weight data type.
        D: Number of channels.
        W: Kernel width.
        has_bias: Whether the filter weights carry a bias.
        silu: Whether SiLU activation is applied.
        xstride: Distance in elements between consecutive token rows of the input, or None.
        wmajor: Weight layout, "time" for packed weights or "channel" for the caller's own
            bias-free weights read in place.
        vec: Number of adjacent channels per tile. Token rows must start on a tile boundary, so
            a row stride that is a multiple of 8 but not of 16 takes tiles of 8.

    Returns:
        Tuple of a dict mapping (single_sequence, macro) to the compiled strip walk, and the
        compiled boundary correction kernel used when a batch holds several sequences.
    """
    walks = {}
    for macro in _MACROS:
        strip = _strip(W, macro)
        walks[True, macro] = V._compile_strip_seq(
            dtype, dtype, D, W, has_bias, silu, strip, vec, 64, macro, xstride, wmajor
        )
        walks[False, macro] = F._compile_strip(
            dtype,
            dtype,
            D,
            W,
            has_bias,
            silu,
            strip,
            vec,
            64,
            True,
            False,
            False,
            False,
            xstride,
            macro,
            wmajor,
        )
    fix = V._compile_boundary_vec(
        dtype, dtype, D, W, has_bias, silu, xstride, vec, _block_size(D // vec), wmajor
    )
    return (walks, fix)


def _strip(W, macro):
    """Return the strip length used with a macro count.

    Three tokens, or more where needed for a thread's span of strip * macro tokens to reach past
    the first width - 1 rows: the thread that starts a sequence discards its outputs for those rows
    into the last row of its span, which must therefore be a row that it also computes properly.
    """
    return max(3, -(-W // macro))


def _seq_config(T, W):
    """Pick the strip length, block size and macro count for a single packed sequence.

    Short strips walked several times keep the last width - 1 tokens in registers, which avoids
    loading them again from rows that collide in the cache when tokens are a power of two apart.

    Args:
        T: Number of tokens.
        W: Kernel width.

    Returns:
        Tuple of strip length, thread block size and macro count.
    """
    macro = 1 if T < 512 else 2 if T < 1024 else 4 if T < 2048 else 8 if T < 4096 else 6
    return (_strip(W, macro), 64, macro)


def _int32(index, device, nseq=1):
    """Return slot indices as int32, defaulting to one slot per sequence in order."""
    if index is None:
        return torch.arange(nseq, device=device, dtype=torch.int32)
    return index if index.dtype == torch.int32 else index.to(torch.int32)


def _flags(flags, device, nseq=1):
    """Return per-sequence initial-state flags as uint8, defaulting to no initial state."""
    if flags is None:
        return torch.zeros(nseq, device=device, dtype=torch.uint8)
    return flags.view(torch.uint8) if flags.dtype == torch.bool else flags.to(torch.uint8)


def _block_size(tiles):
    """Pick a thread block size that divides the tile count when one exists.

    Args:
        tiles: Number of channel tiles per sequence.

    Returns:
        The thread block size for a tiled kernel.
    """
    for bs in (32, 64, 16, 128):
        if tiles % bs == 0:
            return bs
    return 32


def _indexed_config(steps, batch):
    """Pick the tile width and thread block size of the indexed decode kernel.

    These kernels finish within a fraction of a microsecond of the fixed cost of a launch, so their
    latency is the latency of one thread. Few sequences leave the device mostly idle, and threads that
    handle two or four channels finish sooner than threads that handle sixteen; with more
    sequences the wider tile, which needs fewer memory transactions per channel, takes over.
    Measured on B300 for 1 to 48 sequences.

    Args:
        steps: Number of tokens per sequence.
        batch: Number of sequences.

    Returns:
        Tuple of channels per tile and thread block size.
    """
    if steps == 1:
        return (2, 64) if batch <= 8 else (2, 128) if batch <= 16 else (16, 32)
    return (4, 32) if batch <= 8 else (16, 32)


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    intermediate_conv_window=None,
    intermediate_state_indices=None,
):
    """Advance the conv state by one or more tokens and compute their outputs.

    With x of shape (batch, dim) this is one decode step. With x of shape (batch, dim, steps) it
    is the multi-token step that speculative decoding issues when the target model verifies a
    chain of draft tokens: the conv state advances past all steps tokens, and when
    intermediate_conv_window is given the state after every token is recorded there so that the
    caller can restore the state of any accepted prefix.

    If conv_state_indices is None and conv_state has matching batch dimension contiguously, a
    single-token call runs the direct update kernel; otherwise the indexed kernel is launched.

    Args:
        x: Input tensor of shape (batch, dim), or (batch, dim, steps) viewing contiguous (batch,
            steps, dim) storage.
        conv_state: Conv state tensor of shape (slots, dim, width - 1), updated in place.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: Optional activation: "silu", "swish", or None.
        conv_state_indices: Optional int32 tensor of shape (batch,) mapping batch indices to conv
            state slot indices.
        pad_slot_id: Slot index representing padded entries to skip during conv state updates.
        intermediate_conv_window: Optional per-token state cache of shape (slots, steps, dim,
            width - 1), either contiguous or the overlapping view in which the state after token
            t starts t elements into a per-channel row of steps + width - 2 elements.
        intermediate_state_indices: Optional int32 tensor of shape (batch,) mapping batch indices
            to slots of intermediate_conv_window. Defaults to conv_state_indices.

    Returns:
        Output tensor with the shape and strides of x.

    Raises:
        ValueError: If the shapes of x, weight, bias and conv_state are inconsistent.
    """
    _check(x, weight, bias, 1)
    D, W = weight.shape
    if conv_state.dim() != 3 or conv_state.shape[1] != D or conv_state.shape[2] < W - 1:
        raise ValueError(
            f"conv_state shape {tuple(conv_state.shape)} is not (slots, {D}, >= {W - 1})"
        )
    why = update_reason(x, conv_state, weight, bias, intermediate_conv_window)
    if why is not None:
        FB.warn(why)
        return FB.update(
            x,
            conv_state,
            weight,
            bias,
            _act(activation),
            conv_state_indices,
            pad_slot_id,
            intermediate_conv_window,
            intermediate_state_indices,
        )
    return _update(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        pad_slot_id,
        intermediate_conv_window,
        intermediate_state_indices,
    )


def _update(
    x,
    conv_state,
    weight,
    bias,
    activation,
    conv_state_indices,
    pad_slot_id,
    intermediate_conv_window,
    intermediate_state_indices,
):
    """Run causal_conv1d_update on arguments that update_reason() has already accepted."""
    D, W = weight.shape
    act = _act(activation)
    layer = _layer(weight, bias, act)
    B = x.shape[0]
    single = x.dim() == 2
    if (
        single
        and conv_state_indices is None
        and conv_state.shape[0] == B
        and conv_state.is_contiguous()
        and x.is_contiguous()
    ):
        return layer.update(x, conv_state, out=torch.empty_like(x))
    if conv_state_indices is None:
        conv_state_indices = torch.arange(B, device=x.device, dtype=torch.int32)
    elif conv_state_indices.dtype != torch.int32:
        conv_state_indices = conv_state_indices.to(torch.int32)
    L = 1 if single else x.shape[2]
    rows = x if single else x.transpose(1, 2).reshape(B * L, D)
    state2d = conv_state.view(conv_state.shape[0], -1)
    if not (_tiles_ok(rows) and _tiles_ok(state2d)):
        x = x.contiguous()
        out = torch.empty_like(x)
        k = V._compile_update_indexed(
            x.dtype, weight.dtype, D, W, bias is not None, act is not None
        )
        k(x, conv_state, layer.wp, out, conv_state_indices, int(pad_slot_id), B)
        return out
    out = torch.empty((B * L, D), dtype=x.dtype, device=x.device)
    inter = intermediate_conv_window
    layout = None if inter is None else _inter_layout(inter, D, W, L, x.dtype)
    if layout == "dedup":
        inter2d = inter.as_strided((inter.shape[0], D * (L + W - 2)), (inter.stride(0), 1))
    elif layout == "dense":
        inter2d = inter.view(inter.shape[0] * L, D * (W - 1))
    else:
        inter2d = state2d
    iidx = intermediate_state_indices
    if iidx is None:
        iidx = conv_state_indices
    elif iidx.dtype != torch.int32:
        iidx = iidx.to(torch.int32)
    wflat = layer.wflat
    k = V._compile_update_indexed_vec(
        x.dtype,
        weight.dtype,
        D,
        W,
        bias is not None,
        act is not None,
        L,
        layout,
        *_indexed_config(L, B),
        1,
        "time" if wflat is None else "channel",
    )
    weights = layer.wtm if wflat is None else wflat
    k(rows, state2d, weights, out, conv_state_indices, inter2d, iidx, int(pad_slot_id), B)
    return out if single else out.view(B, L, D).transpose(1, 2)


def causal_conv1d_update_ring(
    x,
    ring_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
):
    """Advance a ring conv state by one token and compute the output.

    A ring conv state has shape (slots, width - 1, dim) and is not shifted: the step overwrites the
    row that holds the oldest input, row cache_seqlens % (width - 1), and leaves the others in
    place. At large batch this moves a quarter fewer bytes than the ordered state for width 4. Use
    to_ring and from_ring to convert from and to the (slots, dim, width - 1) state that prefill
    writes.

    Args:
        x: Input tensor of shape (batch, dim).
        ring_state: Ring conv state of shape (slots, width - 1, dim), contiguous, updated in place.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: Optional activation: "silu", "swish", or None.
        cache_seqlens: Int32 tensor of shape (batch,) with the number of tokens each sequence had
            absorbed before this step. The caller increments it afterwards.
        conv_state_indices: Optional int32 tensor of shape (batch,) mapping batch rows to slots.
        pad_slot_id: Slot index marking batch rows to skip.

    Returns:
        Output tensor of shape (batch, dim).

    Raises:
        ValueError: If the shapes of x, weight, bias and ring_state are inconsistent or
            cache_seqlens is missing.
    """
    _check(x, weight, bias, 1)
    D, W = weight.shape
    if x.dim() != 2 or ring_state.dim() != 3 or tuple(ring_state.shape[1:]) != (W - 1, D):
        raise ValueError(
            f"need x (batch, {D}) and ring_state (slots, {W - 1}, {D}), got {tuple(x.shape)} and"
            f" {tuple(ring_state.shape)}"
        )
    if cache_seqlens is None or cache_seqlens.shape != (x.shape[0],):
        raise ValueError("cache_seqlens must be an integer tensor of shape (batch,)")
    act = _act(activation)
    why = _reason(x, weight, bias)
    if why is None and not (ring_state.is_contiguous() and ring_state.dtype == x.dtype):
        why = "the ring state must be contiguous and share the input dtype"
    if why is not None:
        FB.warn(why)
        slots = (
            torch.arange(x.shape[0], device=x.device)
            if conv_state_indices is None
            else conv_state_indices.long()
        )
        live = slots != pad_slot_id
        ordered = R.from_ring(ring_state[slots[live]], cache_seqlens[live])
        out = torch.empty_like(x)
        out[live] = FB.update(x[live], ordered, weight, bias, act, None, pad_slot_id, None, None)
        ring_state[slots[live], (cache_seqlens[live].long() % (W - 1))] = x[live]
        return out
    if cache_seqlens.dtype != torch.int32:
        cache_seqlens = cache_seqlens.to(torch.int32)
    if conv_state_indices is not None and conv_state_indices.dtype != torch.int32:
        conv_state_indices = conv_state_indices.to(torch.int32)
    x = x if x.is_contiguous() else x.contiguous()
    return _layer(weight, bias, act).update_ring(
        x, ring_state, cache_seqlens, conv_state_indices, pad_slot_id
    )
