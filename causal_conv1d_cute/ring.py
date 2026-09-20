"""Single-token decode on a ring conv state.

The usual conv state holds the last width - 1 inputs of every channel in order, so each decode step
rewrites all of it: width - 2 elements move down and one is appended. A ring state keeps the same
inputs in a (slots, width - 1, dim) tensor whose rows are not ordered by age. A step overwrites the
row that holds the oldest input and leaves the others in place, which removes width - 2 of the
width - 1 state writes. Decode at large batch is bound by the bytes it moves, and this is the only
change that moves fewer of them.

Which row is the oldest follows from the number of tokens a sequence has absorbed: row
cache_seqlens % (width - 1). The filter has to be applied in age order. A thread loads the packed
weights of its channels, which do not depend on the position, at the same time as the state rows and
the sequence length, and then picks the weight of every state row in registers by comparing the
position against each possible value. Looking the weights up by position instead would make the
weight load wait for the sequence length load, which costs a second memory round trip at the batch
sizes where decode latency is the latency of one thread. Every thread touches each cache line once:
width - 1 state rows, one input tile and the weights are loaded, then one state row and one output
tile are stored.

to_ring and from_ring convert between this layout and the (slots, dim, width - 1) state that
prefill writes.
"""

import functools
import math
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from .fwd import _DTYPES, packed_width, silu_f32


def to_ring(conv_state, cache_seqlens):
    """Convert an ordered conv state to a ring state.

    Args:
        conv_state: Tensor of shape (slots, dim, width - 1) holding the last width - 1 inputs of
            every channel, oldest first.
        cache_seqlens: Integer tensor of shape (slots,) with the number of tokens each sequence
            has absorbed.

    Returns:
        Tensor of shape (slots, width - 1, dim) in which row (cache_seqlens + k) % (width - 1)
        holds input k of the ordered state.
    """
    K = conv_state.shape[2]
    k = torch.arange(K, device=conv_state.device)
    age = (k[None, :] - cache_seqlens[:, None].long()) % K
    return conv_state.transpose(1, 2).gather(1, age[:, :, None].expand(-1, -1, conv_state.shape[1]))


def from_ring(ring_state, cache_seqlens):
    """Convert a ring state to an ordered conv state.

    Args:
        ring_state: Tensor of shape (slots, width - 1, dim).
        cache_seqlens: Integer tensor of shape (slots,) with the number of tokens each sequence
            has absorbed.

    Returns:
        Tensor of shape (slots, dim, width - 1) holding the inputs oldest first.
    """
    K = ring_state.shape[1]
    k = torch.arange(K, device=ring_state.device)
    row = (cache_seqlens[:, None].long() + k[None, :]) % K
    ordered = ring_state.gather(1, row[:, :, None].expand(-1, -1, ring_state.shape[2]))
    return ordered.transpose(1, 2).contiguous()


class ConvUpdateRing:
    """Single-token decode kernel for a ring conv state, cv adjacent channels per thread."""

    def __init__(self, dim, width, has_bias, silu, cv=4, bs=128, indexed=False):
        """Initialize the kernel parameters.

        Args:
            dim: Number of channels.
            width: Kernel width.
            has_bias: Whether the packed weights carry a bias.
            silu: Whether to apply SiLU activation to outputs.
            cv: Number of adjacent channels per thread.
            bs: Thread block size.
            indexed: Whether batch rows are mapped to state slots through an index tensor.
        """
        assert dim % cv == 0
        self.dim, self.width, self.has_bias, self.silu = (dim, width, has_bias, silu)
        self.cv, self.bs, self.indexed = (cv, bs, indexed)
        self.K = width - 1
        self.P = packed_width(width, False)
        self.tw = math.gcd(cv * self.P, 16)
        self.G = dim // cv
        self.guard = self.G % bs != 0

    @cute.jit
    def _load(self, tile):
        """Load a tensor tile into registers using autovectorized copy.

        Args:
            tile: Tensor tile to load.

        Returns:
            Register tensor containing the loaded tile.
        """
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(self, x, state, wp, b, out, seqlens, idx, n: Int32, pad: Int32, nblocks, stream):
        """Launch the kernel on a CUDA stream.

        Args:
            x: Input activations of shape (batch, dim).
            state: Ring conv state viewed as (slots * (width - 1), dim), one row per state row.
            wp: Packed filter weights of shape (1, dim * P), without the bias.
            b: Bias of shape (1, dim). Ignored unless has_bias.
            out: Output activations of shape (batch, dim).
            seqlens: Number of tokens each sequence has absorbed, shape (batch,).
            idx: State slot of each batch row, shape (batch,). Ignored unless indexed.
            n: Number of threads that have work, batch * (dim // cv).
            pad: Slot index marking batch rows to skip. Ignored unless indexed.
            nblocks: Number of thread blocks.
            stream: CUDA stream for launch.
        """
        self.kernel(x, state, wp, b, out, seqlens, idx, n, pad).launch(
            grid=[nblocks, 1, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.jit
    def _body(self, x, state, wp, b, out, seqlens, slot, bi, g):
        """Compute one tile of one batch row and overwrite its oldest state row.

        State row j holds the input of age (j - pos) % (width - 1), counted from the oldest, so
        its weight is that entry of the channel's filter. The entry is selected in registers from
        the width - 1 candidates.

        Args:
            x: Input activations of shape (batch, dim).
            state: Ring conv state, one row per state row.
            wp: Packed filter weights.
            b: Bias.
            out: Output activations of shape (batch, dim).
            seqlens: Number of tokens each sequence has absorbed.
            slot: State slot of this batch row.
            bi: Batch row.
            g: Tile index along the channel dimension.
        """
        W: cutlass.Constexpr = self.width
        K: cutlass.Constexpr = self.K
        P: cutlass.Constexpr = self.P
        CV: cutlass.Constexpr = self.cv
        TW: cutlass.Constexpr = self.tw
        NW: cutlass.Constexpr = self.cv * self.P // self.tw
        xt = cute.logical_divide(x, (None, CV))
        ot = cute.logical_divide(out, (None, CV))
        st = cute.logical_divide(state, (None, CV))
        wt = cute.logical_divide(wp, (None, TW))
        pos = seqlens[bi] % K
        base = slot * K
        sr = [self._load(st[base + j, (None, g)]) for j in range(K)]
        wr = [self._load(wt[0, (None, g * NW + m)]) for m in range(NW)]
        xr = self._load(xt[bi, (None, g)])
        br = (
            self._load(cute.logical_divide(b, (None, CV))[0, (None, g)])
            if const_expr(self.has_bias)
            else None
        )
        orr = cute.make_rmem_tensor_like(xr)
        for c in cutlass.range_constexpr(CV):
            a = br[c].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
            for j in cutlass.range_constexpr(K):
                q = c * P + (j - (K - 1)) % K
                wv = wr[q // TW][q % TW]
                for p in cutlass.range_constexpr(K - 1):
                    q = c * P + (j - (K - 2 - p)) % K
                    wv = wr[q // TW][q % TW] if pos == K - 2 - p else wv
                a = a + sr[j][c].to(Float32) * wv.to(Float32)
            a = a + xr[c].to(Float32) * wr[(c * P + K) // TW][(c * P + K) % TW].to(Float32)
            if const_expr(self.silu):
                a = silu_f32(a, out.element_type.width == 32)
            orr[c] = a.to(out.element_type)
        cute.autovec_copy(xr, st[base + pos, (None, g)])
        cute.autovec_copy(orr, ot[bi, (None, g)])

    @cute.kernel
    def kernel(self, x, state, wp, b, out, seqlens, idx, n: Int32, pad: Int32):
        """Map a thread to a batch row and a channel tile.

        Args:
            x: Input activations of shape (batch, dim).
            state: Ring conv state, one row per state row.
            wp: Packed filter weights.
            b: Bias.
            out: Output activations of shape (batch, dim).
            seqlens: Number of tokens each sequence has absorbed.
            idx: State slot of each batch row.
            n: Number of threads that have work.
            pad: Slot index marking batch rows to skip.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * self.bs + tidx
        if const_expr(self.indexed):
            last = n - 1
            ic = i if i < last else last
            bi = ic // self.G
            slot = idx[bi]
            if (i < n) & (slot != pad):
                self._body(x, state, wp, b, out, seqlens, slot, bi, ic - bi * self.G)
        else:
            if const_expr(self.guard):
                if i < n:
                    bi = i // self.G
                    self._body(x, state, wp, b, out, seqlens, bi, bi, i - bi * self.G)
            else:
                bi = i // self.G
                self._body(x, state, wp, b, out, seqlens, bi, bi, i - bi * self.G)


def _rows(dtype, cols, align):
    """Create a fake two-dimensional tensor with a symbolic row count and row stride.

    Args:
        dtype: Element data type.
        cols: Number of columns.
        align: Number of elements that rows are aligned to.

    Returns:
        Fake tensor descriptor for compilation.
    """
    return cute.runtime.make_fake_tensor(
        dtype,
        (cute.sym_int(), cols),
        stride=(cute.sym_int64(divisibility=align), 1),
        assumed_align=align * dtype.width // 8,
    )


@functools.cache
def _compile_ring(dtype, wdtype, dim, width, has_bias, silu, cv, bs, indexed):
    """Compile the ring decode kernel for given configurations.

    Args:
        dtype: Activation data type.
        wdtype: Weight data type.
        dim: Channel dimension.
        width: Kernel width.
        has_bias: Whether the packed weights carry a bias.
        silu: Whether SiLU activation is enabled.
        cv: Number of adjacent channels per thread.
        bs: Thread block size.
        indexed: Whether batch rows are mapped to state slots through an index tensor.

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    P = packed_width(width, False)
    ints = lambda: cute.runtime.make_fake_tensor(
        cutlass.Int32, (cute.sym_int(),), stride=(1,), assumed_align=4
    )
    return cute.compile(
        ConvUpdateRing(dim, width, has_bias, silu, cv, bs, indexed),
        _rows(dt, dim, cv),
        _rows(dt, dim, cv),
        _rows(wt, dim * P, math.gcd(cv * P, 16)),
        _rows(wt, dim, cv),
        _rows(dt, dim, cv),
        ints(),
        ints(),
        Int32(1),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_update_ring(
    x,
    ring_state,
    weight_packed,
    bias,
    width,
    cache_seqlens,
    activation=None,
    conv_state_indices=None,
    pad_slot_id=-1,
    out=None,
    cv=4,
    bs=128,
):
    """Advance a ring conv state by one token and compute the output.

    Args:
        x: Input tensor of shape (batch, dim), contiguous.
        ring_state: Ring conv state of shape (slots, width - 1, dim), contiguous, updated in
            place.
        weight_packed: Packed filter weights of shape (dim, P) from pack_weight(weight, None).
        bias: Optional bias tensor of shape (dim,), loaded separately from the weights so that
            the weights of a tile stay within one load.
        width: Kernel width.
        cache_seqlens: Int32 tensor of shape (batch,) with the number of tokens each sequence had
            absorbed before this step. The caller increments it afterwards.
        activation: Activation function to apply, either "silu" or None.
        conv_state_indices: Optional int32 tensor of shape (batch,) mapping batch rows to state
            slots. Without it batch row b uses slot b.
        pad_slot_id: Slot index marking batch rows to skip.
        out: Optional output tensor of shape (batch, dim).
        cv: Number of adjacent channels per thread.
        bs: Thread block size.

    Returns:
        Output tensor of shape (batch, dim).
    """
    B, D = x.shape
    K = width - 1
    assert ring_state.shape[1:] == (K, D) and ring_state.is_contiguous() and x.is_contiguous()
    if out is None:
        out = torch.empty_like(x)
    indexed = conv_state_indices is not None
    k = _compile_ring(
        x.dtype,
        weight_packed.dtype,
        D,
        width,
        bias is not None,
        activation is not None,
        cv,
        bs,
        indexed,
    )
    n = B * (D // cv)
    k(
        x,
        ring_state.view(-1, D),
        weight_packed.view(1, -1),
        (bias if bias is not None else weight_packed.view(-1)[:D]).view(1, D),
        out,
        cache_seqlens,
        conv_state_indices if indexed else cache_seqlens,
        n,
        int(pad_slot_id),
        (n + bs - 1) // bs,
    )
    return out
