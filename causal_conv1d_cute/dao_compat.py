"""Serve calls written against the Dao-AILab causal_conv1d interface with the CuTe DSL kernels.

Many model implementations, among them mamba_ssm, flash-linear-attention and the fast paths of
Hugging Face transformers, call ``causal_conv1d.causal_conv1d_fn`` and
``causal_conv1d.causal_conv1d_update``. install() wraps both functions and rebinds them in every
module that has already imported them, so those models pick up the kernels without a code change.

The wrappers keep the upstream signatures. A call is passed on to the upstream function whenever it
asks for something the kernels do not provide: gradients, seq_idx, initial_states,
return_final_states, a circular conv state through cache_seqlens, a multi-token update, a conv
state longer than width - 1, or tensors the kernels cannot serve. stats counts both outcomes.
"""

import torch
from . import _patch
from . import functional as fn

TARGETS_FN = ("causal_conv1d.causal_conv1d_interface.causal_conv1d_fn",)
TARGETS_UPDATE = ("causal_conv1d.causal_conv1d_interface.causal_conv1d_update",)
stats = {"fn_fast": 0, "fn_fallback": 0, "update_fast": 0, "update_fallback": 0}


def _needs_grad(*tensors):
    """Return whether autograd would record an operation on any of the tensors."""
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)


def around_fn(
    original_fn,
    x,
    weight,
    bias=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """Serve a causal_conv1d_fn call, or pass it to the upstream function.

    Args:
        original_fn: The upstream causal_conv1d_fn.
        x: Input tensor of shape (batch, dim, seqlen), contiguous or viewing (batch, seqlen, dim)
            storage.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        seq_idx: Optional sequence index tensor. Passes the call on.
        initial_states: Optional initial conv states. Passes the call on.
        return_final_states: Whether to return final conv states. Passes the call on when true.
        final_states_out: Optional output tensor for the final conv states.
        activation: Activation function name: "silu", "swish", or None.

    Returns:
        Output tensor of shape (batch, dim, seqlen).
    """
    served = (
        seq_idx is None
        and initial_states is None
        and not return_final_states
        and activation in (None, "silu", "swish")
        and x.dim() == 3
        and not _needs_grad(x, weight, bias)
        and fn.supported(x, weight, bias)
    )
    if not served:
        stats["fn_fallback"] += 1
        return original_fn(
            x,
            weight,
            bias,
            seq_idx=seq_idx,
            initial_states=initial_states,
            return_final_states=return_final_states,
            final_states_out=final_states_out,
            activation=activation,
        )
    stats["fn_fast"] += 1
    return fn.causal_conv1d_fn(x, weight, bias, activation)


def around_update(
    original_fn,
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
):
    """Serve a single-token causal_conv1d_update call, or pass it to the upstream function.

    Args:
        original_fn: The upstream causal_conv1d_update.
        x: Input tensor of shape (batch, dim). A (batch, dim, seqlen) input passes the call on.
        conv_state: Conv state tensor of shape (batch, dim, width - 1), updated in place.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: Activation function name: "silu", "swish", or None.
        cache_seqlens: Optional positions of a circular conv state. Passes the call on.
        conv_state_indices: Optional int32 tensor of shape (batch,) selecting rows of conv_state.

    Returns:
        Output tensor of shape (batch, dim).
    """
    served = (
        cache_seqlens is None
        and activation in (None, "silu", "swish")
        and x.dim() == 2
        and not _needs_grad(x, weight, bias)
        and fn.update_reason(x, conv_state, weight, bias) is None
    )
    if not served:
        stats["update_fallback"] += 1
        return original_fn(
            x,
            conv_state,
            weight,
            bias,
            activation=activation,
            cache_seqlens=cache_seqlens,
            conv_state_indices=conv_state_indices,
        )
    stats["update_fast"] += 1
    return fn._update(
        x, conv_state, weight, bias, activation, conv_state_indices, fn.PAD_SLOT_ID, None, None
    )


def install():
    """Wrap the upstream causal_conv1d functions in the current process.

    Import the model code first: only modules that are already loaded are rebound. Calling
    install() again is a no-op.

    Returns:
        The number of rebound module attributes.
    """
    count = 0
    for target in TARGETS_FN:
        count += _patch.wrap(target, around_fn)
    for target in TARGETS_UPDATE:
        count += _patch.wrap(target, around_update)
    return count
