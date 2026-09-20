"""Dispatch causal 1D convolution forward and update kernels.

Selects kernel implementations and launch configurations from a tuned configuration table based on
tensor shape, layout, and layer parameters. Falls back to heuristics when an exact match is not
found.
"""

import functools
import json
import torch
from pathlib import Path
from . import fwd as F
from . import ring as R
from . import update as U

_TABLE = Path(__file__).with_name("tuned.json")


def arch():
    """Return the compute architecture of the current CUDA device, such as "sm_103"."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


@functools.cache
def _table():
    """Load and cache the tuned kernel configurations measured on the current architecture.

    The table on disk is keyed by architecture first. Configurations measured on one architecture
    are never applied to another; a device without an entry uses the heuristics.

    Returns:
        dict: Mapping from configuration keys to lists of tuned kernel launch parameters, or an
            empty dictionary if nothing was measured on this architecture.
    """
    full = json.loads(_TABLE.read_text()) if _TABLE.exists() else {}
    return full.get(arch(), {})


def _key(kind, layout, D, W, bias, act):
    """Construct a lookup key for the tuned configuration table.

    Args:
        kind: Kernel kind string, either "fwd" or "update".
        layout: Memory layout string ("bdl", "btd", "std", or "pad").
        D: Number of channels.
        W: Kernel width.
        bias: Whether bias is present.
        act: Activation function name or None.

    Returns:
        Formatted string key identifying the layer configuration.
    """
    return f"{kind}|{layout}|{D}|{W}|{int(bias)}|{act or 'none'}"


def fwd_candidates(layout, W, B, L):
    """Enumerate valid forward kernel configurations for an input shape.

    For layout "bdl", candidate configurations include row-major and ragged kernels based on
    sequence length divisibility and kernel width bounds. For layout "btd", candidate configurations
    include channel-last kernels across single-element and chunked sequence lengths.

    Args:
        layout: Memory layout, either "bdl" (batch, channels, sequence length) or "btd" (batch,
            sequence length, channels).
        W: Kernel width.
        B: Batch size.
        L: Sequence length.

    Returns:
        List of configuration tuples valid for the given dimensions.
    """
    c = []
    if layout == "bdl":
        if L % 16 == 0:
            c += [("row", 16, 256), ("row", 16, 128)]
            c += [("row", 16, bs, 2) for bs in (64, 128, 256) if L % 32 == 0]
        if L % 8 == 0 and W - 1 <= 8:
            c += [("row", 8, 256), ("row", 8, 128)]
        if L >= 16:
            c += [
                ("ragged", 16, 256),
                ("ragged", 16, 128),
                ("ragged2", 16, 256),
                ("ragged2", 16, 128),
            ]
        if L >= 8 and W - 1 <= 8:
            c += [("ragged", 8, 256), ("ragged", 8, 128)]
    else:
        c += [("cl", 1, 64), ("cl", 1, 128)]
        c += [("strip", S, 64) for S in (2, 3, 4, 5, 6, 7, 8, 10, 12) if L >= S]
        c += [("strip", S, 128) for S in (2, 4, 5, 7, 8) if L >= S]
        c += [("strip", S, 64, M) for S in (2, 3, 4, 5) for M in (2, 4, 6, 8) if L >= S * M]
    return c


def _heuristic_fwd(layout, W, B, L):
    """Select a default forward kernel configuration using heuristics.

    For layout "bdl", selects a row-major kernel if the sequence length is divisible by 16 or 8,
    otherwise selects a ragged kernel. For layout "btd", selects a channel-last kernel based on
    sequence length and the total token count (B * L).

    Args:
        layout: Memory layout, either "bdl" or "btd".
        W: Kernel width.
        B: Batch size.
        L: Sequence length.

    Returns:
        Kernel configuration tuple (kernel_type, vector_size_or_chunk, block_size).
    """
    if layout == "bdl":
        if L % 16 == 0:
            return ("row", 16, 256)
        if L % 8 == 0 and W <= 9:
            return ("row", 8, 256)
        return ("ragged2", 16, 256) if L >= 128 else ("ragged", 16, 256)
    work = B * L
    if work <= 64 or L < 2:
        return ("cl", 1, 64)
    return (
        ("strip", 2, 64)
        if work <= 512 and L >= 2
        else ("strip", 5, 64)
        if L >= 5
        else ("strip", 2, 64)
    )


def _heuristic_update(layout, D, B):
    """Select a decode kernel configuration for a shape that was not tuned.

    Decode finishes close to the fixed cost of a launch, so threads must stay short: two channels per thread
    for up to eight sequences, four beyond. Wider tiles need fewer memory transactions per channel
    but make each thread longer, and measured slower at every batch size on B300.

    Args:
        layout: Conv state layout, either "std" or "pad".
        D: Number of channels.
        B: Batch size.

    Returns:
        Kernel configuration tuple (kernel_type, channels_per_thread, block_size).
    """
    if layout == "std":
        return ("v1", 2, 128) if B <= 8 else ("v1", 4, 128)
    if B <= 8 or D // 4 % 128 != 0:
        return ("pad", 0, 128)
    return ("padv", 4, 128)


def _lookup(kind, layout, D, W, bias, act, B, L, valid):
    """Find the best matching kernel configuration from the tuned table.

    Args:
        kind: Kernel kind string, either "fwd" or "update".
        layout: Memory layout string ("bdl", "btd", "std", or "pad").
        D: Number of channels.
        W: Kernel width.
        bias: Whether bias is present.
        act: Activation function name or None.
        B: Batch size.
        L: Sequence length.
        valid: Collection of allowed configuration tuples, or None.

    Returns:
        Configuration tuple with minimal relative distance in total work (B * L),
        or None if no match is found.
    """
    rows = _table().get(_key(kind, layout, D, W, bias, act), [])
    best = None
    for r in rows:
        cfg = tuple(r["cfg"])
        if valid is not None and cfg not in valid:
            continue
        d = abs(r["B"] * r["L"] - B * L) / max(B * L, 1) + (
            0 if (r["B"], r["L"]) == (B, L) else 1e-06
        )
        if best is None or d < best[0]:
            best = (d, cfg)
    return best[1] if best else None


class CausalConv1d:
    """Manage weights and kernel dispatch for causal 1D convolution.

    Holds filter weights formatted for row-major forward kernels, time-major channel-last forward
    kernels, and padded conv state decode update kernels.
    """

    def __init__(self, weight, bias=None, activation=None):
        """Initialize and pre-pack weights for causal 1D convolution.

        Args:
            weight: Filter weights tensor of shape (dim, width).
            bias: Optional bias tensor of shape (dim,).
            activation: Activation function name, either None, "silu", or "swish".

        Raises:
            ValueError: If activation is not an accepted value, or weight does not have shape
                (dim, width) with dim a multiple of 16 and width between 2 and 8.
        """
        if activation not in (None, "silu", "swish"):
            raise ValueError(f'activation must be None, "silu" or "swish", got {activation!r}')
        if weight.dim() != 2 or not 2 <= weight.shape[1] <= 8 or weight.shape[0] % 16 != 0:
            raise ValueError(
                f"weight must have shape (dim, width) with dim a multiple of 16 and 2 <= width <= 8,"
                f" got {tuple(weight.shape)}"
            )
        self.D, self.W = weight.shape
        self.has_bias, self.act = (bias is not None, None if activation is None else "silu")
        self.weight, self.bias = (weight, bias)
        self._packed = {}
        self._plans = {}
        self._vplans = {}
        self._wpnb = None

    def _pack(self, name, build):
        """Return a packed copy of the weights, building it on first use.

        The copies are built lazily because a caller may need none of them: the serving kernels
        read bias-free weights in place, and inside a serving engine an allocation that the engine
        does not make itself moves every buffer allocated after it.
        """
        if name not in self._packed:
            self._packed[name] = build()
        return self._packed[name]

    @property
    def wp(self):
        """Packed (dim, P) filter weights, bias in the last column."""
        return self._pack("wp", lambda: F.pack_weight(self.weight, self.bias))

    @property
    def wtm(self):
        """Time-major (width + int(has_bias), dim) filter weights."""
        return self._pack("wtm", lambda: F.pack_weight_timemajor(self.weight, self.bias))

    @property
    def wpad(self):
        """Filter weights padded to four columns for the padded conv state."""
        return self._pack("wpad", lambda: U.pad_weight(self.weight))

    @property
    def wflat(self):
        """The caller's own (dim, width) weights as one (1, dim * width) row, or None.

        None when the weights cannot be read in place: they carry a bias, which the in-place
        kernels do not take, or they are not contiguous or not aligned to 16 elements.
        """
        w = self.weight
        ok = self.bias is None and w.is_contiguous() and w.data_ptr() % (16 * w.element_size()) == 0
        return w.view(1, -1) if ok else None

    def fwd(self, x, out=None, cfg=None):
        """Execute the causal 1D convolution forward pass.

        Args:
            x: Input tensor of shape (batch, dim, seqlen). Must be contiguous for row-major
                execution, or a transposed view of contiguous (batch, seqlen, dim) storage for
                channel-last execution.
            out: Optional output tensor of shape (batch, dim, seqlen). Must not share underlying
                storage with x.
            cfg: Optional kernel launch configuration tuple. If None, chosen from the tuned table or
                heuristics.

        Returns:
            Output tensor of shape (batch, dim, seqlen) after convolution and
            optional bias and activation.

        Raises:
            ValueError: If out shares underlying storage with x. In place execution is prohibited
                because overlapping thread reads and stores create data races under aliased storage.
        """
        B, D, L = x.shape
        if out is not None and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
            raise ValueError(
                "causal_conv1d_cute prefill is out-of-place only: `out` must not share storage with `x`"
            )
        layout = "bdl" if x.is_contiguous() else "btd"
        if cfg is None:
            strided = layout == "btd" and x.stride(2) != D
            plan = (layout, strided, B, L)
            cfg = self._plans.get(plan)
            if cfg is None:
                valid = fwd_candidates(layout, self.W, B, L)
                if strided:
                    valid = [c for c in valid if c[0] == "strip"]
                cfg = _lookup(
                    "fwd", layout, D, self.W, self.has_bias, self.act, B, L, valid
                ) or _heuristic_fwd(layout, self.W, B, L)
                if strided and cfg[0] != "strip":
                    cfg = ("strip", 2, 64)
                if len(self._plans) < 4096:
                    self._plans[plan] = cfg
        kind, a, bs, *rest = cfg
        if kind == "row":
            return F.causal_conv1d_fwd_rowmajor(
                x,
                self.wp,
                self.W,
                self.has_bias,
                self.act,
                out=out,
                vec=a,
                bs=bs,
                macro=rest[0] if rest else 1,
            )
        if kind in ("ragged", "ragged2"):
            return F.causal_conv1d_fwd_rowmajor_ragged(
                x,
                self.wp,
                self.W,
                self.has_bias,
                self.act,
                out=out,
                vec=a,
                bs=bs,
                two_pass=kind == "ragged2",
            )
        if kind == "cl":
            return F.causal_conv1d_fwd_channellast(
                x, self.wtm, self.W, self.has_bias, self.act, out=out, bs=bs
            )
        return F.causal_conv1d_fwd_channellast_strip(
            x,
            self.wtm,
            self.W,
            self.has_bias,
            self.act,
            out=out,
            strip=a,
            bs=bs,
            hoist=True,
            macro=rest[0] if rest else 1,
        )

    def _wp_nobias(self):
        """Return the packed filter weights without the bias, packing them on first use."""
        if self._wpnb is None:
            self._wpnb = F.pack_weight(self.weight, None)
        return self._wpnb

    def update_ring(
        self,
        x,
        ring_state,
        cache_seqlens,
        conv_state_indices=None,
        pad_slot_id=-1,
        out=None,
        cfg=None,
    ):
        """Execute a single-step decode update on a ring conv state.

        Args:
            x: Input tensor of shape (batch, dim), contiguous.
            ring_state: Ring conv state of shape (slots, width - 1, dim), updated in place. Row
                cache_seqlens % (width - 1) of a slot holds its oldest input and is overwritten.
            cache_seqlens: Int32 tensor of shape (batch,) with the number of tokens each sequence
                had absorbed before this step.
            conv_state_indices: Optional int32 tensor of shape (batch,) mapping batch rows to
                state slots.
            pad_slot_id: Slot index marking batch rows to skip.
            out: Optional output tensor of shape (batch, dim).
            cfg: Optional kernel launch configuration tuple ("ring", channels_per_thread,
                block_size). If None, chosen from the tuned table or heuristics.

        Returns:
            Output tensor of shape (batch, dim).
        """
        B = x.shape[0]
        if cfg is None:
            cfg = _lookup(
                "update", "ring", self.D, self.W, self.has_bias, self.act, B, 1, None
            ) or ("ring", 2 if B <= 2 else 4, 128)
        return R.causal_conv1d_update_ring(
            x,
            ring_state,
            self.wp if self.bias is None else self._wp_nobias(),
            self.bias,
            self.W,
            cache_seqlens,
            self.act,
            conv_state_indices,
            pad_slot_id,
            out,
            cv=cfg[1],
            bs=cfg[2],
        )

    @staticmethod
    def update_candidates(layout, D=None):
        """Enumerate valid kernel configurations for decode state update.

        For padded layouts, vectorized configurations update state in place without boundary guards.
        These configurations are valid only when the thread count equals the work count, satisfying
        (D // cv) % bs == 0.

        Args:
            layout: Conv state layout, either "std" or "pad".
            D: Number of channels, or None.

        Returns:
            List of configuration tuples valid for the given layout and channel count.
        """
        sizes = (32, 64, 128, 256)
        if layout == "ring":
            return [("ring", cv, bs) for cv in (2, 4, 8, 16) for bs in (64, 128, 256)]
        if layout == "std":
            c = [("v5", 0, bs) for bs in (64, 128, 256)] + [("v0", 0, 128)]
            return c + [("v1", cv, bs) for cv in (2, 4, 8, 16) for bs in sizes]
        c = [("pad", 0, bs) for bs in (64, 128, 256)]
        padv = [("padv", cv, bs) for cv in (2, 4, 8, 16) for bs in sizes]
        return c + [v for v in padv if D is None or (D % v[1] == 0 and D // v[1] % v[2] == 0)]

    def update(self, x, conv_state, out=None, cfg=None):
        """Execute a single-step decode update for causal 1D convolution.

        Args:
            x: Input tensor of shape (batch, dim) for the current timestep.
            conv_state: Conv state tensor of shape (batch, dim, width - 1) for standard layout, or
                (batch, dim, padded_len) with padded_len a power of two >= width for padded layout.
                Updated in place.
            out: Optional output tensor of shape (batch, dim).
            cfg: Optional kernel launch configuration tuple. If None, chosen from the tuned table or
                fallback heuristics.

        Returns:
            Output tensor of shape (batch, dim) after convolving the state with the
            current input and applying optional bias and activation.
        """
        B, D = x.shape
        P = self.wpad.shape[1]
        layout = "pad" if conv_state.shape[-1] == P and P != self.W - 1 else "std"
        if cfg is None:
            cfg = _lookup(
                "update",
                layout,
                D,
                self.W,
                self.has_bias,
                self.act,
                B,
                1,
                self.update_candidates(layout, D),
            )
            if cfg is None:
                cfg = _heuristic_update(layout, D, B)
        kind, cv, bs, *rest = cfg
        if kind == "v5":
            return U.causal_conv1d_update_std2d(
                x, conv_state, self.wp, self.W, self.has_bias, self.act, out=out, bs=bs
            )
        if kind == "v0":
            return U.causal_conv1d_update(
                x, conv_state, self.weight, self.bias, self.act, out=out, bs=bs
            )
        if kind == "v1":
            return U.causal_conv1d_update_vec(
                x,
                conv_state,
                self.weight,
                self.bias,
                self.act,
                out=out,
                cv=cv,
                bs=bs,
                tiles=rest[0] if rest else 1,
            )
        if kind == "pad":
            return U.causal_conv1d_update_padded(
                x, conv_state, self.wpad, self.W, self.bias, self.act, out=out, bs=bs
            )
        return U.causal_conv1d_update_padded_vec(
            x, conv_state, self.wpad, self.W, self.bias, self.act, out=out, cv=cv, bs=bs
        )
