"""Serve sglang causal conv1d calls with the CuTe DSL kernels.

sglang reaches its conv kernels through two functions: causal_conv1d_fn for prefill over a
variable-length (packed) batch, and causal_conv1d_update for decode, including the multi-token
step that verifies speculative drafts. This module wraps both. A call that the kernels can serve
runs on them; any other call is passed unchanged to the function that was wrapped, so installing
the wrappers never removes functionality.

Decoding without speculation does not go through causal_conv1d_update: sglang calls one fused
kernel that reads the conv input from the [Q | K | V | Z] projection row, updates the conv state,
and copies Z, B and A into tensors of their own. That function is wrapped as well and served by a
kernel that does the same in one launch.

There are two ways to install them. Inside a server, sglang discovers the entry point plugin()
in every scheduler process; it does nothing unless the environment variable
CAUSAL_CONV1D_CUTE_SGLANG is set to 1. Inside a single process, call install() after importing
sglang.

A prefill call with one sequence runs as a single kernel launch; several sequences take two, one
over the packed stream and one over the sequence boundaries. Calls with fewer than
MIN_PREFILL_TOKENS tokens are passed on.

Outputs are allocated out of place. The first call of every distinct kind is logged together
with the reason when it was passed on, and stats counts both outcomes.
"""

import logging
import os
import pkgutil
import torch
from . import _patch
from . import functional as fn
from . import unpack

PAD_SLOT_ID = fn.PAD_SLOT_ID
ENV_ENABLE = "CAUSAL_CONV1D_CUTE_SGLANG"
ENV_PATHS = "CAUSAL_CONV1D_CUTE_SGLANG_PATHS"
ENV_DRY_RUN = "CAUSAL_CONV1D_CUTE_SGLANG_DRY_RUN"
MIN_PREFILL_TOKENS = 16
TARGETS_FN = ("sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_fn",)
TARGETS_UPDATE = (
    "sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_update",
    "sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_update",
)
TARGETS_UNPACK = (
    "sglang.kernels.ops.attention.triton_gdn_fused_proj.fused_qkvzba_causal_conv1d_update_contiguous",
)
TARGET_LOAD = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"
stats = {
    "fn_fast": 0,
    "fn_fallback": 0,
    "update_fast": 0,
    "update_fallback": 0,
    "unpack_fast": 0,
    "unpack_fallback": 0,
}
logger = logging.getLogger(__name__)
_seen = set()
_UNSERVED = (
    "cache_seqlens",
    "num_accept_tokens",
    "retrieve_next_token",
    "retrieve_next_sibling",
    "retrieve_parent_token",
)


def _note(op, why, x):
    """Count one call and log the first call of its kind.

    Args:
        op: Operation name: "fn", "update" or "unpack".
        why: Reason the call was passed to the wrapped function, or None when it was served.
        x: Input tensor of the call.
    """
    stats[f"{op}_{'fast' if why is None else 'fallback'}"] += 1
    key = (op, why, x.dim(), x.dtype)
    if key not in _seen and len(_seen) < 64:
        _seen.add(key)
        logger.info(
            "causal_conv1d_%s %s: input %s strides %s %s%s",
            op,
            "served" if why is None else "passed on",
            tuple(x.shape),
            x.stride(),
            x.dtype,
            "" if why is None else f" ({why})",
        )


def around_fn(
    original_fn,
    x,
    weight,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    conv_states=None,
    activation="silu",
    pad_slot_id=PAD_SLOT_ID,
    **kwargs,
):
    """Serve a prefill call, or pass it to the wrapped function.

    Args:
        original_fn: The wrapped sglang function.
        x: Packed input tensor of shape (dim, total_tokens).
        weight: Filter weights tensor of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        query_start_loc: Sequence boundary tensor of shape (batch + 1,), dtype torch.int32.
        cache_indices: Optional mapping of shape (batch,) from sequence index to conv state slot.
        has_initial_state: Optional boolean tensor of shape (batch,) indicating whether to read
            the initial conv state of each sequence.
        conv_states: Optional conv state buffer of shape (slots, dim, width - 1), updated in place.
        activation: Activation function name: "silu", "swish", or None.
        pad_slot_id: Slot index in cache_indices marking sequences to skip. Their outputs are
            undefined.
        **kwargs: Additional keyword arguments, forwarded when the call is passed on.

    Returns:
        Output tensor of shape (dim, total_tokens), allocated out of place.
    """
    if activation not in (None, "silu", "swish"):
        why = "activation"
    elif x.dim() == 2 and x.shape[1] < MIN_PREFILL_TOKENS:
        why = f"fewer than {MIN_PREFILL_TOKENS} tokens"
    else:
        why = fn.varlen_reason(x, weight, bias, query_start_loc, conv_states)
    if why is None and _dry_run():
        why = "dry run"
    _note("fn", why, x)
    if why is not None:
        return original_fn(
            x,
            weight,
            bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            conv_states=conv_states,
            activation=activation,
            pad_slot_id=pad_slot_id,
            **kwargs,
        )
    return fn._varlen(
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


def around_update(
    original_fn,
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    *args,
    conv_state_indices=None,
    intermediate_conv_window=None,
    intermediate_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    **kwargs,
):
    """Serve a decode call, or pass it to the wrapped function.

    Single-token calls and the multi-token calls that verify a chain of speculative draft tokens
    are served. Calls that use a circular conv state, a tree of draft tokens, or a pre-accepted
    token count are passed on.

    Args:
        original_fn: The wrapped sglang function.
        x: Input tensor of shape (batch, dim) or (batch, dim, steps).
        conv_state: Conv state tensor of shape (slots, dim, width - 1), updated in place.
        weight: Filter weights tensor of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: Activation function name: "silu", "swish", a bool, or None.
        *args: Further positional arguments; their presence passes the call on.
        conv_state_indices: Optional mapping of shape (batch,) from batch row to conv state slot.
        intermediate_conv_window: Optional cache of shape (slots, steps, dim, width - 1) that
            receives the conv state after every token.
        intermediate_state_indices: Optional mapping of shape (batch,) from batch row to slot of
            intermediate_conv_window.
        pad_slot_id: Slot index in conv_state_indices marking rows to skip.
        **kwargs: Additional keyword arguments, forwarded when the call is passed on.

    Returns:
        Output tensor with the shape of x.
    """
    why = None
    if args:
        why = "positional arguments after activation"
    elif any(kwargs.get(k) is not None for k in _UNSERVED):
        why = "uses " + ", ".join(k for k in _UNSERVED if kwargs.get(k) is not None)
    elif isinstance(activation, str) and activation not in ("silu", "swish"):
        why = "activation"
    else:
        why = fn.update_reason(x, conv_state, weight, bias, intermediate_conv_window)
    if why is None and _dry_run():
        why = "dry run"
    _note("update", why, x)
    if why is not None:
        return original_fn(
            x,
            conv_state,
            weight,
            bias,
            activation,
            *args,
            conv_state_indices=conv_state_indices,
            pad_slot_id=pad_slot_id,
            **(
                {}
                if intermediate_conv_window is None
                else {
                    "intermediate_conv_window": intermediate_conv_window,
                    "intermediate_state_indices": intermediate_state_indices,
                }
            ),
            **kwargs,
        )
    if conv_state_indices is None:
        conv_state_indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int32)
    return fn._update(
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


def around_unpack(
    original_fn,
    mixed_qkvz,
    mixed_ba,
    conv_state,
    conv_weight,
    conv_bias,
    conv_state_indices,
    *,
    qkv_dim,
    v_dim,
    num_v_heads,
    head_v_dim,
    activation,
    pad_slot_id=PAD_SLOT_ID,
):
    """Serve the fused decode call of a Gated DeltaNet layer, or pass it to the wrapped function.

    Args:
        original_fn: The wrapped sglang function.
        mixed_qkvz: Projection rows of shape (batch, qkv_dim + v_dim) laid out as [Q | K | V | Z].
        mixed_ba: Projection rows of shape (batch, 2 * num_v_heads) laid out as [B | A].
        conv_state: Conv state tensor of shape (slots, qkv_dim, width - 1), updated in place.
        conv_weight: Filter weights tensor of shape (qkv_dim, width).
        conv_bias: Optional bias tensor of shape (qkv_dim,).
        conv_state_indices: Mapping of shape (batch,) from batch row to conv state slot.
        qkv_dim: Number of conv channels.
        v_dim: Width of Z.
        num_v_heads: Width of B and of A.
        head_v_dim: Width of one head of Z.
        activation: Activation function name: "silu", "swish" or None.
        pad_slot_id: Slot index in conv_state_indices marking rows to skip.

    Returns:
        Tuple of the conv outputs of shape (batch, qkv_dim), Z of shape (batch, num_v_heads,
        head_v_dim), B of shape (batch, num_v_heads) and A of shape (batch, num_v_heads).
    """
    why = None
    if activation not in (None, "silu", "swish"):
        why = "activation"
    elif conv_weight.dim() != 2 or mixed_qkvz.dim() != 2 or mixed_ba.dim() != 2:
        why = "tensor ranks"
    elif mixed_qkvz.shape[1] != qkv_dim + v_dim or v_dim != num_v_heads * head_v_dim:
        why = "projection layout"
    elif conv_weight.shape[0] != qkv_dim or conv_state_indices.shape[0] != mixed_qkvz.shape[0]:
        why = "shapes"
    else:
        why = unpack.unpack_reason(
            mixed_qkvz, mixed_ba, conv_state, conv_weight, conv_bias, num_v_heads
        )
    if why is None and _dry_run():
        why = "dry run"
    _note("unpack", why, mixed_qkvz)
    if why is not None:
        return original_fn(
            mixed_qkvz,
            mixed_ba,
            conv_state,
            conv_weight,
            conv_bias,
            conv_state_indices,
            qkv_dim=qkv_dim,
            v_dim=v_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            activation=activation,
            pad_slot_id=pad_slot_id,
        )
    batch = mixed_qkvz.shape[0]
    out, z, b, a = unpack.causal_conv1d_update_unpack(
        mixed_qkvz,
        mixed_ba,
        conv_state,
        conv_weight,
        conv_bias,
        fn._int32(conv_state_indices, mixed_qkvz.device),
        num_v_heads,
        None if activation is None else "silu",
        int(pad_slot_id),
        *unpack.unpack_config(qkv_dim, batch, mixed_qkvz.stride(0)),
    )
    return (out, z.view(batch, num_v_heads, head_v_dim), b, a)


def _dry_run():
    """Return whether every call is to be passed to the wrapped function.

    With CAUSAL_CONV1D_CUTE_SGLANG_DRY_RUN=1 the wrappers are installed and log what they would
    serve, but sglang's own kernels run. A server started this way shows what installing the
    plugin costs or gains before any kernel is exchanged; on one GPU that alone moved the decode
    step time by up to 0.3%, because allocations made by a plugin shift the buffers of the server.
    """
    return os.environ.get(ENV_DRY_RUN, "0") == "1"


def after_load_model(result, runner, *args, **kwargs):
    """Pack the conv weights that cannot be read in place as soon as the model is loaded.

    Bias-free, aligned weights are read in place by the serving kernels and need no copy. Any
    other layer needs a packed copy; left to the first call, that copy would be allocated while
    sglang warms up and captures its CUDA graphs, after it has sized its memory pools, which
    changes where every later buffer of the server lands. Allocated here, next to the model
    weights, the copies leave the rest of the layout as it is without the plugin.

    Args:
        result: Return value of ModelRunner.load_model, passed through.
        runner: The ModelRunner whose model was loaded.
        *args: Unused.
        **kwargs: Unused.

    Returns:
        The unchanged return value of ModelRunner.load_model.
    """
    count = 0
    for module in runner.model.modules():
        weights = getattr(module, "conv_weights", None)
        bias = getattr(module, "bias", None)
        act = getattr(module, "activation", None)
        if isinstance(weights, torch.Tensor):
            weights = (weights,)
        if not isinstance(weights, (tuple, list)):
            continue
        for weight in weights:
            ok = isinstance(weight, torch.Tensor) and weight.is_cuda and weight.dim() == 2
            if ok and 2 <= weight.shape[1] <= 8 and weight.shape[0] % 16 == 0:
                one = bias if isinstance(bias, torch.Tensor) and bias.dim() == 1 else None
                layer = fn._layer(weight, one, None if act is None else "silu")
                if layer.wflat is None:
                    _ = layer.wtm
                    count += 1
    logger.info("causal-conv1d-cute: packed the conv weights of %d layers at load", count)
    return result


def _present(target):
    """Return whether the function or method at a dotted path exists in the installed sglang."""
    try:
        pkgutil.resolve_name(target)
    except (ImportError, AttributeError):
        return False
    return True


def _wrappers():
    """Return the (target, wrapper) pairs selected by CAUSAL_CONV1D_CUTE_SGLANG_PATHS.

    The variable is a comma-separated subset of "fn" (prefill), "update" (decode and speculative
    verify) and "unpack" (fused decode without speculation). All three are selected by default.
    """
    paths = os.environ.get(ENV_PATHS, "fn,update,unpack").split(",")
    table = (("fn", TARGETS_FN, around_fn), ("update", TARGETS_UPDATE, around_update))
    pairs = [(t, hook) for name, targets, hook in table if name in paths for t in targets]
    if "unpack" in paths:
        pairs += [(t, around_unpack) for t in TARGETS_UNPACK if _present(t)]
    return pairs


def plugin():
    """Register the wrappers with sglang's plugin hook registry.

    sglang calls this entry point in every scheduler process. Nothing is registered unless the
    environment variable CAUSAL_CONV1D_CUTE_SGLANG is set to 1.
    """
    if os.environ.get(ENV_ENABLE, "0") != "1":
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    pairs = _wrappers()
    for target, hook in pairs:
        HookRegistry.register(target, hook, HookType.AROUND)
    if pairs and _present(TARGET_LOAD):
        HookRegistry.register(TARGET_LOAD, after_load_model, HookType.AFTER)
    logger.info("causal-conv1d-cute: wrapping %d sglang conv entry points", len(pairs))


def install():
    """Wrap the sglang conv entry points in the current process.

    Replaces each entry point in its defining module and in every loaded module that imported it
    by name. Use this when sglang runs inside the current process, for instance in a test; a server
    loads the wrappers through plugin() in its scheduler processes instead. Calling install() again
    is a no-op.

    Returns:
        The number of rebound module attributes.
    """
    return sum(_patch.wrap(target, hook) for target, hook in _wrappers())
