"""Execute forward causal 1D convolutions using CuTe.

Forward causal 1D convolution kernels specialized by channel count, kernel width, bias, activation,
and data type. Supports row-major (batch, dim, seqlen) and channel-last (batch, seqlen, dim) memory
layouts. Filter weights and optional bias are stored in the same row aligned to a power of two to
allow vector loads per channel. Accumulation is performed in float32 precision.
"""

import functools
import math
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from . import _iket

_DTYPES = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}


@cute.jit
def silu_f32(a, exact: cutlass.Constexpr = False):
    """Compute SiLU activation on float32 values.

    Evaluates x * sigmoid(x) using 0.5 * x * (tanh(0.5 * x) + 1.0). When exact is False, uses
    approximate tanh instructions. Set exact to True when writing float32 outputs.

    Args:
        a: Input float32 value.
        exact: Whether to use exact tanh rather than approximate tanh.

    Returns:
        Resulting float32 value.
    """
    half = Float32(0.5)
    if const_expr(exact):
        return a * (cute.math.tanh(a * half) * half + half)
    return a * (cute.math.tanh(a * half, approx=True) * half + half)


def packed_width(width: int, has_bias: bool) -> int:
    """Compute the aligned row width for packed filter weights.

    Args:
        width: Convolution kernel width.
        has_bias: Whether a bias element is present.

    Returns:
        Smallest power of two capable of holding width weights and optional bias.
    """
    return 1 << (width + int(has_bias) - 1).bit_length()


def pack_weight(weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Pack filter weights and optional bias into aligned channel rows.

    Filter weights and optional bias are stored in the same row aligned to a power of two. Filter
    weights occupy indices 0 to width - 1, and optional bias is placed in the final column of each
    channel row.

    Args:
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).

    Returns:
        Packed weight tensor of shape (dim, packed_width) with bias in the last
        column if provided.
    """
    D, W = weight.shape
    P = packed_width(W, bias is not None)
    wp = torch.zeros(D, P, device=weight.device, dtype=weight.dtype)
    wp[:, :W] = weight
    if bias is not None:
        wp[:, P - 1] = bias.to(weight.dtype)
    return wp


class ConvFwdRowMajor:
    """Execute row-major forward causal 1D convolution.

    Processes contiguous inputs of shape (batch, dim, seqlen) where seqlen is a multiple of the
    vector width. Each thread computes a vector of output elements within a single channel row by
    loading the current tile, preceding tile, and packed filter weights. Causal zero-padding on the
    first tile of each row is applied via selection.

    When fast is True:
    - The thread count equals the work count, eliminating bounds checks where out-of-range threads
      are mapped onto the last element.
    - Data loads are issued before row index arithmetic.
    - Integer division for row indexing is computed using precomputed multipliers and shifts.
    """

    def __init__(self, dim, width, has_bias, silu, vec=16, bs=256, fast=False, macro=1):
        """Initialize row-major forward convolution parameters.

        Args:
            dim: Number of channels.
            width: Convolution kernel width.
            has_bias: Whether packed weights contain channel bias.
            silu: Whether to apply SiLU activation.
            vec: Vector width per thread tile.
            bs: Thread block size.
            fast: Whether to use fast indexing without bounds checks.
            macro: Number of consecutive tiles that one thread handles, one after the other.
        """
        assert width - 1 <= vec
        self.dim, self.width, self.has_bias, self.silu, self.vec, self.bs = (
            dim,
            width,
            has_bias,
            silu,
            vec,
            bs,
        )
        self.P, self.fast, self.M = (packed_width(width, has_bias), fast, macro)

    @cute.jit
    def _load(self, tile):
        """Load a tile from global memory into registers.

        Args:
            tile: Tensor tile to load.

        Returns:
            Register tensor containing loaded values.
        """
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(
        self,
        x,
        wp,
        out,
        tiles_per_row: Int32,
        ntiles: Int32,
        magic: cutlass.Int64,
        shift: cutlass.Int64,
        nblocks: Int32,
        stream,
    ):
        """Launch the row-major forward convolution kernel over a 1D grid.

        Args:
            x: Input tensor in global memory.
            wp: Packed filter weights tensor in global memory.
            out: Output tensor in global memory.
            tiles_per_row: Number of vector tiles per row.
            ntiles: Total number of vector tiles across all rows.
            magic: Magic multiplier for division by tiles_per_row.
            shift: Shift count for division by tiles_per_row.
            nblocks: Number of thread blocks in the grid.
            stream: CUDA stream for execution.
        """
        self.kernel(x, wp, out, tiles_per_row, ntiles, magic, shift).launch(
            grid=[nblocks, 1, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x,
        wp,
        out,
        tiles_per_row: Int32,
        ntiles: Int32,
        magic: cutlass.Int64,
        shift: cutlass.Int64,
    ):
        """Execute row-major forward causal convolution for a single thread.

        Each thread computes a vector of output elements from adjacent input tiles and packed filter
        weights. Reconstructs causal history elements, accumulates products in float32 precision,
        adds optional bias, applies optional SiLU activation, and writes the output tile. When fast
        is False, out-of-range threads are mapped onto the last element.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        W: cutlass.Constexpr = self.width
        V: cutlass.Constexpr = self.vec
        P: cutlass.Constexpr = self.P
        _iket.push("index")
        i = bidx * self.bs + tidx
        if const_expr(not self.fast):
            i = i if i < ntiles else ntiles - 1
        M: cutlass.Constexpr = self.M
        xt = cute.logical_divide(x, (None, V))
        ot = cute.logical_divide(out, (None, V))
        t0 = i * M
        ip = t0 - 1 if t0 > 0 else t0
        _iket.pop()
        _iket.push("loads")
        cur = [self._load(xt[0, (None, t0 + q)]) for q in range(M)]
        prev = self._load(xt[0, (None, ip)])
        if const_expr(self.fast):
            row = ((i.to(cutlass.Int64) * magic) >> shift).to(Int32)
        else:
            row = i // tiles_per_row
        s = i - row * tiles_per_row
        c = row % self.dim
        wr = self._load(cute.logical_divide(wp, (None, P))[c, (None, 0)])
        _iket.pop()
        _iket.push("compute")
        zero = cutlass.Float32(0.0)
        first = s == 0
        look = [zero if first else prev[V - (W - 1) + j].to(Float32) for j in range(W - 1)]
        wf = [wr[k].to(Float32) for k in range(W)]
        outs = []
        for q in cutlass.range_constexpr(M):
            orr = cute.make_rmem_tensor_like(cur[0])
            for e in cutlass.range_constexpr(V):
                a = wr[P - 1].to(Float32) if const_expr(self.has_bias) else zero
                for k in cutlass.range_constexpr(W):
                    j = e - (W - 1) + k
                    if const_expr(j >= 0):
                        a = a + cur[q][j].to(Float32) * wf[k]
                    elif const_expr(q == 0):
                        a = a + look[j + (W - 1)] * wf[k]
                    else:
                        a = a + cur[q - 1][V + j].to(Float32) * wf[k]
                if const_expr(self.silu):
                    a = silu_f32(a, out.element_type.width == 32)
                orr[e] = a.to(out.element_type)
            outs.append(orr)
        _iket.pop()
        _iket.push("store")
        for q in cutlass.range_constexpr(M):
            cute.autovec_copy(outs[q], ot[0, (None, t0 + q)])
        _iket.pop()


@functools.cache
def _compile_rowmajor(dtype, wdtype, dim, width, has_bias, silu, vec, bs, fast=False, macro=1):
    """Compile and cache the row-major forward convolution kernel.

    Args:
        dtype: Data type of input and output tensors.
        wdtype: Data type of packed filter weights.
        dim: Number of channels.
        width: Convolution kernel width.
        has_bias: Whether packed weights include bias.
        silu: Whether to apply SiLU activation.
        vec: Vector width per thread tile.
        bs: Thread block size.
        fast: Whether to compile with bounds-free indexing.
        macro: Number of consecutive tiles of one row that a thread walks.

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    P = packed_width(width, has_bias)
    n = cute.sym_int(divisibility=vec)
    flat = lambda: cute.runtime.make_fake_tensor(
        dt, (1, n), stride=(cute.sym_int64(divisibility=vec), 1), assumed_align=vec * dt.width // 8
    )
    wfake = cute.runtime.make_fake_tensor(
        wt, (dim, P), stride=(P, 1), assumed_align=P * wt.width // 8
    )
    return cute.compile(
        ConvFwdRowMajor(dim, width, has_bias, silu, vec, bs, fast, macro),
        flat(),
        wfake,
        flat(),
        Int32(1),
        Int32(1),
        cutlass.Int64(1),
        cutlass.Int64(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_fwd_rowmajor(
    x, weight_packed, width, has_bias, activation=None, out=None, vec=16, bs=256, fast=None, macro=1
):
    """Compute forward causal 1D convolution on row-major inputs.

    Applies a 1D causal convolution across contiguous row-major inputs where the sequence length is
    divisible by the vector width.

    Args:
        x: Contiguous input tensor of shape (batch, dim, seqlen).
        weight_packed: Packed filter weights of shape (dim, packed_width) produced by pack_weight.
        width: Convolution kernel width.
        has_bias: Whether bias values are present in weight_packed.
        activation: Activation function to apply, either 'silu' or None.
        out: Optional output tensor of shape (batch, dim, seqlen). If None, a new tensor is
            allocated.
        vec: Vector width per thread tile. Default: 16.
        bs: Thread block size. Default: 256.
        fast: Whether to use fast indexing without bounds checks. When None, enabled if the
            thread count is divisible by bs.
        macro: Number of consecutive tiles of one row that a thread walks, loading the preceding
            tile once. Requires seqlen to be a multiple of vec * macro. Default: 1.

    Returns:
        Output tensor of shape (batch, dim, seqlen).
    """
    B, D, L = x.shape
    assert x.is_contiguous() and L % (vec * macro) == 0
    if out is None:
        out = torch.empty_like(x)
    tpr = L // (vec * macro)
    ntiles = B * D * tpr
    if fast is None:
        fast = ntiles % bs == 0
    assert not fast or ntiles % bs == 0
    k = _compile_rowmajor(
        x.dtype,
        weight_packed.dtype,
        D,
        width,
        has_bias,
        activation is not None,
        vec,
        bs,
        fast,
        macro,
    )
    shift = 32 + max(tpr - 1, 0).bit_length()
    k(
        x.view(1, -1),
        weight_packed,
        out.view(1, -1),
        tpr,
        ntiles,
        (1 << shift) // tpr + 1,
        shift,
        (ntiles + bs - 1) // bs,
    )
    return out


def pack_weight_timemajor(weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Pack filter weights and optional bias for channel-last layouts.

    Rearranges filter weights into time-major format so that row k contains position k across all
    channels, with an optional final row containing the channel bias.

    Args:
        weight: Filter weight tensor of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).

    Returns:
        Contiguous tensor of shape (width + int(has_bias), dim).
    """
    rows = [weight.t().contiguous()]
    if bias is not None:
        rows.append(bias.to(weight.dtype).view(1, -1))
    return torch.cat(rows, dim=0).contiguous()


class ConvFwdChannelLast:
    """Execute channel-last forward causal 1D convolution.

    Processes inputs stored in (batch, seqlen, dim) order. Assigns one thread to an output tile of
    channels for a single token. Each thread loads kernel width input tiles across preceding tokens
    for the same channel tile, together with filter weights and optional bias, before writing the
    output tile. Out-of-range threads are mapped onto the last element and redundantly write
    identical values without divergent branches.

    Supports two thread arrangements:
    - 'C': Adjacent threads process adjacent channel tiles of the same token.
    - 'T': Consecutive threads process consecutive tokens of the same channel tile.
    """

    def __init__(self, dim, width, has_bias, silu, vec=16, bs=256, arrangement="C", exact=False):
        """Initialize channel-last forward convolution parameters.

        Args:
            dim: Number of channels.
            width: Convolution kernel width.
            has_bias: Whether time-major weights contain channel bias.
            silu: Whether to apply SiLU activation.
            vec: Channel vector width per thread tile.
            bs: Thread block size.
            arrangement: Thread mapping order, either 'C' or 'T'.
            exact: Whether the thread count equals the work count.
        """
        assert dim % vec == 0 and arrangement in ("C", "T")
        self.dim, self.width, self.has_bias, self.silu = (dim, width, has_bias, silu)
        self.vec, self.bs, self.arr, self.G, self.exact = (vec, bs, arrangement, dim // vec, exact)

    @cute.jit
    def _load(self, tile):
        """Load a tile from global memory into registers.

        Args:
            tile: Tensor tile to load.

        Returns:
            Register tensor containing loaded values.
        """
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(self, x, wtm, out, seqlen: Int32, nblocks: Int32, nbatch: Int32, stream):
        """Launch the channel-last forward convolution kernel over a 2D grid.

        Args:
            x: Input tensor in global memory.
            wtm: Time-major filter weights tensor in global memory.
            out: Output tensor in global memory.
            seqlen: Sequence length of each batch sequence.
            nblocks: Number of thread blocks along grid dimension x.
            nbatch: Batch size along grid dimension y.
            stream: CUDA stream for execution.
        """
        self.kernel(x, wtm, out, seqlen).launch(
            grid=[nblocks, nbatch, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, wtm, out, seqlen: Int32):
        """Execute channel-last forward causal convolution for a single thread.

        Loads input tiles across preceding tokens for a channel group alongside time-major filter
        weights and optional bias. Masks inputs for token indices prior to zero, accumulates in
        float32 precision, applies optional SiLU activation, and writes the output vector tile. When
        exact is False, out-of-range threads are mapped onto the last element.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, b, _ = cute.arch.block_idx()
        W: cutlass.Constexpr = self.width
        V: cutlass.Constexpr = self.vec
        G: cutlass.Constexpr = self.G
        i = bidx * self.bs + tidx
        if const_expr(not self.exact):
            last = seqlen * G - 1
            i = i if i < last else last
        t = i // G
        j = i - t * G
        if const_expr(self.arr == "T"):
            j = i // seqlen
            t = i - j * seqlen
        base = b * seqlen
        xt = cute.logical_divide(x, (None, V))
        wt = cute.logical_divide(wtm, (None, V))
        zero = cutlass.Float32(0.0)
        data = []
        for k in cutlass.range_constexpr(W):
            lag = W - 1 - k
            tl = t - lag if t >= lag else t
            data.append(self._load(xt[0, (None, (base + tl) * G + j)]))
        wts = [self._load(wt[k, (None, j)]) for k in range(W)]
        bias = self._load(wt[W, (None, j)]) if const_expr(self.has_bias) else None
        orr = cute.make_rmem_tensor_like(data[0])
        for e in cutlass.range_constexpr(V):
            a = bias[e].to(Float32) if const_expr(self.has_bias) else zero
            for k in cutlass.range_constexpr(W):
                lag = W - 1 - k
                if const_expr(lag == 0):
                    a = a + data[k][e].to(Float32) * wts[k][e].to(Float32)
                else:
                    xv = data[k][e].to(Float32) if t >= lag else zero
                    a = a + xv * wts[k][e].to(Float32)
            if const_expr(self.silu):
                a = silu_f32(a, out.element_type.width == 32)
            orr[e] = a.to(out.element_type)
        cute.autovec_copy(orr, cute.logical_divide(out, (None, V))[0, (None, (base + t) * G + j)])


@functools.cache
def _compile_channellast(
    dtype, wdtype, dim, width, has_bias, silu, vec, bs, arrangement, exact=False
):
    """Compile and cache the channel-last forward convolution kernel.

    Args:
        dtype: Data type of input and output tensors.
        wdtype: Data type of time-major filter weights.
        dim: Number of channels.
        width: Convolution kernel width.
        has_bias: Whether weights include bias.
        silu: Whether to apply SiLU activation.
        vec: Channel vector width per thread tile.
        bs: Thread block size.
        arrangement: Thread mapping order, either 'C' or 'T'.
        exact: Whether the thread count equals the work count.

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    n = cute.sym_int(divisibility=dim)
    flat = lambda: cute.runtime.make_fake_tensor(
        dt, (1, n), stride=(cute.sym_int64(divisibility=vec), 1), assumed_align=vec * dt.width // 8
    )
    rows = width + int(has_bias)
    wfake = cute.runtime.make_fake_tensor(
        wt, (rows, dim), stride=(dim, 1), assumed_align=vec * wt.width // 8
    )
    return cute.compile(
        ConvFwdChannelLast(dim, width, has_bias, silu, vec, bs, arrangement, exact),
        flat(),
        wfake,
        flat(),
        Int32(1),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_fwd_channellast(
    x, weight_tm, width, has_bias, activation=None, out=None, vec=16, bs=256, arrangement="C"
):
    """Compute forward causal 1D convolution on channel-last inputs.

    Applies causal convolution to inputs stored contiguously in channel-last layout (batch, seqlen,
    dim).

    Args:
        x: Input tensor of shape (batch, dim, seqlen) viewing contiguous (batch, seqlen, dim)
            storage.
        weight_tm: Time-major packed weight tensor of shape (width + int(has_bias), dim).
        width: Convolution kernel width.
        has_bias: Whether bias values are present in weight_tm.
        activation: Activation function to apply, either 'silu' or None.
        out: Optional output tensor of shape (batch, dim, seqlen) viewing contiguous (batch, seqlen,
            dim) memory. If None, a new tensor is allocated.
        vec: Channel vector width per thread tile. Default: 16.
        bs: Thread block size. Default: 256.
        arrangement: Thread mapping order, either 'C' (channel-first) or 'T' (token-first). Default:
            'C'.

    Returns:
        Output tensor of shape (batch, dim, seqlen).
    """
    B, D, L = x.shape
    xs = x.transpose(1, 2)
    assert xs.is_contiguous(), "channel-last kernel needs (B, L, D)-contiguous storage"
    if out is None:
        out = torch.empty_like(xs).transpose(1, 2)
    tiles = L * (D // vec)
    k = _compile_channellast(
        x.dtype,
        weight_tm.dtype,
        D,
        width,
        has_bias,
        activation is not None,
        vec,
        bs,
        arrangement,
        tiles % bs == 0,
    )
    k(
        xs.reshape(1, -1),
        weight_tm,
        out.transpose(1, 2).reshape(1, -1),
        L,
        (tiles + bs - 1) // bs,
        B,
    )
    return out


def tile_weights(kernel, w, j):
    """Load the filter weights of channel tile j of a kernel.

    Args:
        kernel: Kernel object with width, vec, has_bias, wmajor and a _load method.
        w: Filter weights. For wmajor "time" a (width + int(has_bias), dim) tensor, one row per
            filter position and a last row for the bias. For wmajor "channel" the caller's own
            (dim, width) tensor viewed as (1, dim * width), which has no bias.
        j: Tile index along the channel dimension.

    Returns:
        Tuple (wr, br). wr[k][e] is the weight of filter position k for channel e of the tile, in
        the weight data type; br is a register tensor holding the bias of the tile, or None.
    """
    W, V = (kernel.width, kernel.vec)
    if kernel.wmajor == "time":
        wt = cute.logical_divide(w, (None, V))
        wr = [kernel._load(wt[k, (None, j)]) for k in range(W)]
        return (wr, kernel._load(wt[W, (None, j)]) if kernel.has_bias else None)
    span = math.gcd(V * W, 16)
    count = V * W // span
    wt = cute.logical_divide(w, (None, span))
    raw = [kernel._load(wt[0, (None, j * count + m)]) for m in range(count)]
    wr = [[raw[(e * W + k) // span][(e * W + k) % span] for e in range(V)] for k in range(W)]
    return (wr, None)


def fake_weights(wdtype, dim, width, has_bias, vec, wmajor):
    """Create the fake weights tensor that tile_weights expects.

    Args:
        wdtype: Weight data type.
        dim: Number of channels.
        width: Kernel width.
        has_bias: Whether the time-major weights carry a trailing bias row.
        vec: Number of adjacent channels per tile.
        wmajor: Weight layout, "time" or "channel".

    Returns:
        Fake tensor descriptor for compilation.
    """
    if wmajor == "time":
        return cute.runtime.make_fake_tensor(
            wdtype,
            (width + int(has_bias), dim),
            stride=(dim, 1),
            assumed_align=vec * wdtype.width // 8,
        )
    span = math.gcd(vec * width, 16)
    return cute.runtime.make_fake_tensor(
        wdtype, (1, dim * width), stride=(dim * width, 1), assumed_align=span * wdtype.width // 8
    )


class ConvFwdChannelLastStrip:
    """Execute channel-last forward causal conv1d using token strips.

    Processes channel-last inputs where each thread computes a strip of tokens across a vector of
    channels. Loads data tiles for preceding tokens and strip tokens together with time-major filter
    weights and optional bias, storing the resulting output tiles. Out-of-range threads are mapped
    onto the last element and store duplicate values. Causal zero-padding for indices prior to token
    0 is applied using register selection. When sequence length is not a multiple of strip size, the
    final strip shifts backward to terminate at the sequence boundary.
    """

    def __init__(
        self,
        dim,
        width,
        has_bias,
        silu,
        strip=16,
        vec=16,
        bs=256,
        hoist=False,
        probe=False,
        exact=False,
        aligned=False,
        xstride=None,
        macro=1,
        wmajor="time",
    ):
        """Initialize token-strip channel-last convolution parameters.

        Args:
            dim: Number of channels.
            width: Convolution kernel width.
            has_bias: Whether time-major weights contain channel bias.
            silu: Whether to apply SiLU activation.
            strip: Number of tokens per strip processed by each thread.
            vec: Channel vector width per thread tile.
            bs: Thread block size.
            hoist: Whether all loads of a strip are issued before any store.
            probe: Whether to skip the arithmetic and copy the input to the output, which measures
                the cost of the memory transactions alone.
            exact: Whether the thread count equals the work count.
            aligned: Whether sequence length is a multiple of strip size.
            xstride: Distance in elements between consecutive token rows of the input. Defaults to
                dim. A larger value reads the leading dim entries of each row of a wider buffer.
            macro: Number of consecutive strips that one thread walks. The thread keeps the last
                width - 1 tokens in registers between strips, so those tokens are loaded once per
                thread instead of once per strip. Requires hoist.
            wmajor: Weight layout, "time" for packed time-major weights or "channel" for the
                caller's own bias-free (dim, width) weights read in place.
        """
        assert dim % vec == 0 and wmajor in ("time", "channel")
        assert not (wmajor == "channel" and has_bias)
        self.wmajor = wmajor
        self.dim, self.width, self.has_bias, self.silu = (dim, width, has_bias, silu)
        self.S, self.vec, self.bs, self.G = (strip, vec, bs, dim // vec)
        self.hoist, self.probe = (hoist, probe)
        self.exact, self.aligned = (exact, aligned)
        self.XG = (xstride or dim) // vec
        self.M = macro
        self.seq1 = False
        assert (xstride or dim) % vec == 0 and self.XG >= self.G
        assert macro == 1 or hoist

    @cute.jit
    def _load(self, tile):
        """Load a tile from global memory into registers.

        Args:
            tile: Tensor tile to load.

        Returns:
            Register tensor containing loaded values.
        """
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(
        self, x, wtm, out, seqlen: Int32, nstrips: Int32, nblocks: Int32, nbatch: Int32, stream
    ):
        """Launch the token-strip channel-last forward kernel over a grid.

        Args:
            x: Input tensor in global memory.
            wtm: Time-major filter weights tensor in global memory.
            out: Output tensor in global memory.
            seqlen: Sequence length of each batch sequence.
            nstrips: Number of token strips per sequence.
            nblocks: Number of thread blocks along grid dimension x.
            nbatch: Batch size along grid dimension y.
            stream: CUDA stream for execution.
        """
        self.kernel(x, wtm, out, seqlen, nstrips).launch(
            grid=[nblocks, nbatch, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.jit
    def _walk(self, x, wtm, out, base, t0, j, scratch):
        """Convolve macro * strip consecutive tokens of one channel tile, starting at token t0.

        Args:
            x: Flat view of the input.
            wtm: Time-major filter weights.
            out: Flat view of the output.
            base: Index of the first token of the sequence that t0 counts from.
            t0: First token of the span, relative to base.
            j: Tile index along the channel dimension.
            scratch: Output row that receives the first width - 1 outputs of a sequence when the
                kernel maintains a conv state; those outputs are computed elsewhere from the old
                state. The caller passes the last row of this thread's span, which the thread
                stores after every other row, so the discarded values never survive and the
                output needs no extra row. Redirecting the store keeps the walk free of branches.
                Unused otherwise.
        """
        W: cutlass.Constexpr = self.width
        V: cutlass.Constexpr = self.vec
        G: cutlass.Constexpr = self.G
        XG: cutlass.Constexpr = self.XG
        S: cutlass.Constexpr = self.S
        M: cutlass.Constexpr = self.M
        xt = cute.logical_divide(x, (None, V))
        ot = cute.logical_divide(out, (None, V))
        zero = cutlass.Float32(0.0)
        _iket.push("loads")
        wr, bias = tile_weights(self, wtm, j)
        wts = [[wr[k][e].to(Float32) for e in range(V)] for k in range(W)]
        b32 = [bias[e].to(Float32) for e in range(V)] if const_expr(self.has_bias) else None
        win = []
        for n in cutlass.range_constexpr(W - 1):
            lag = W - 1 - n
            ok = t0 >= lag
            tl = t0 - lag if ok else t0
            r = self._load(xt[0, (None, (base + tl) * XG + j)])
            win.append([r[e].to(Float32) if ok else zero for e in range(V)])
        if const_expr(self.hoist):
            cur = [self._load(xt[0, (None, (base + t0 + m) * XG + j)]) for m in range(S)]
            for m in cutlass.range_constexpr(S):
                win.append([cur[m][e].to(Float32) for e in range(V)])
        _iket.pop()
        _iket.push("compute")
        outs = []
        for m in cutlass.range_constexpr(S):
            if const_expr(not self.hoist):
                r = self._load(xt[0, (None, (base + t0 + m) * XG + j)])
                win.append([r[e].to(Float32) for e in range(V)])
            orr = cute.make_rmem_tensor_like(ot[0, (None, j)])
            for e in cutlass.range_constexpr(V):
                a = b32[e] if const_expr(self.has_bias) else zero
                if const_expr(self.probe):
                    a = win[m + W - 1][e]
                else:
                    for k in cutlass.range_constexpr(W):
                        a = a + win[m + k][e] * wts[k][e]
                    if const_expr(self.silu):
                        a = silu_f32(a, out.element_type.width == 32)
                orr[e] = a.to(out.element_type)
            if const_expr(self.hoist):
                outs.append(orr)
            else:
                cute.autovec_copy(orr, ot[0, (None, (base + t0 + m) * G + j)])
        _iket.pop()
        _iket.push("stores")
        if const_expr(self.hoist):
            for m in cutlass.range_constexpr(S):
                if const_expr(self.seq1 and m < W - 1):
                    row = t0 + m
                    dst = row if row >= W - 1 else scratch
                    cute.autovec_copy(outs[m], ot[0, (None, dst * G + j)])
                else:
                    cute.autovec_copy(outs[m], ot[0, (None, (base + t0 + m) * G + j)])
        _iket.pop()
        for q in cutlass.range_constexpr(1, M):
            nxt = [self._load(xt[0, (None, (base + t0 + q * S + m) * XG + j)]) for m in range(S)]
            for m in cutlass.range_constexpr(S):
                win.append([nxt[m][e].to(Float32) for e in range(V)])
            more = []
            for m in cutlass.range_constexpr(S):
                orr = cute.make_rmem_tensor_like(nxt[0])
                for e in cutlass.range_constexpr(V):
                    a = b32[e] if const_expr(self.has_bias) else zero
                    for k in cutlass.range_constexpr(W):
                        a = a + win[q * S + m + k][e] * wts[k][e]
                    if const_expr(self.silu):
                        a = silu_f32(a, out.element_type.width == 32)
                    orr[e] = a.to(out.element_type)
                more.append(orr)
            for m in cutlass.range_constexpr(S):
                cute.autovec_copy(more[m], ot[0, (None, (base + t0 + q * S + m) * G + j)])

    @cute.kernel
    def kernel(self, x, wtm, out, seqlen: Int32, nstrips: Int32):
        """Execute token-strip channel-last convolution for a single thread.

        Loads historical tokens, mapping indices onto the boundary and zero-masking out-of-bounds
        positions. Accumulates across the token strip using register values and time-major filter
        weights. When all loads are issued before any store, reads all input data tiles before
        writing output tiles. When exact is False, out-of-range threads are mapped onto the last
        element.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, b, _ = cute.arch.block_idx()
        W: cutlass.Constexpr = self.width
        V: cutlass.Constexpr = self.vec
        G: cutlass.Constexpr = self.G
        XG: cutlass.Constexpr = self.XG
        S: cutlass.Constexpr = self.S
        M: cutlass.Constexpr = self.M
        i = bidx * self.bs + tidx
        if const_expr(not self.exact):
            last = nstrips * G - 1
            i = i if i < last else last
        sidx = i // G
        j = i - sidx * G
        t0 = sidx * (S * M)
        if const_expr(not self.aligned):
            tmax = seqlen - S * M
            t0 = t0 if t0 < tmax else tmax
        self._walk(x, wtm, out, b * seqlen, t0, j, Int32(0))


@functools.cache
def _compile_strip(
    dtype,
    wdtype,
    dim,
    width,
    has_bias,
    silu,
    strip,
    vec,
    bs,
    hoist=False,
    probe=False,
    exact=False,
    aligned=False,
    xstride=None,
    macro=1,
    wmajor="time",
):
    """Compile and cache the token-strip channel-last convolution kernel.

    Args:
        dtype: Data type of input and output tensors.
        wdtype: Data type of time-major filter weights.
        dim: Number of channels.
        width: Convolution kernel width.
        has_bias: Whether weights include bias.
        silu: Whether to apply SiLU activation.
        strip: Number of tokens per strip processed by each thread.
        vec: Channel vector width per thread tile.
        bs: Thread block size.
        hoist: Whether all loads of a strip are issued before any store.
        probe: Whether to skip the arithmetic and copy the input to the output.
        exact: Whether the thread count equals the work count.
        aligned: Whether sequence length is a multiple of strip size.
        xstride: Distance in elements between consecutive token rows of the input, or None when
            the input rows are dense.
        macro: Number of consecutive strips that one thread walks.
        wmajor: Weight layout, "time" or "channel".

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    flat = lambda div: cute.runtime.make_fake_tensor(
        dt,
        (1, cute.sym_int(divisibility=div)),
        stride=(cute.sym_int64(divisibility=vec), 1),
        assumed_align=vec * dt.width // 8,
    )
    wfake = fake_weights(wt, dim, width, has_bias, vec, wmajor)
    return cute.compile(
        ConvFwdChannelLastStrip(
            dim,
            width,
            has_bias,
            silu,
            strip,
            vec,
            bs,
            hoist,
            probe,
            exact,
            aligned,
            xstride,
            macro,
            wmajor,
        ),
        flat(vec if xstride else dim),
        wfake,
        flat(dim),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def channellast_row_stride(x, vec=16):
    """Return the token row stride of a row-strided channel-last input, or None.

    A row-strided channel-last input is a (batch, dim, seqlen) view whose channels are adjacent in
    memory and whose token rows are spaced more than dim elements apart, such as the leading dim
    columns of a wider projection output.

    Args:
        x: Input tensor of shape (batch, dim, seqlen).
        vec: Channel vector width that the row stride must be a multiple of.

    Returns:
        The row stride in elements if x has that layout and is not dense, otherwise None.
    """
    B, D, L = x.shape
    sb, sd, st = x.stride()
    if sd != 1 or st <= D or st % vec != 0 or x.data_ptr() % (vec * x.element_size()) != 0:
        return None
    if B != 1 and sb != L * st:
        return None
    return st


def causal_conv1d_fwd_channellast_strip(
    x,
    weight_tm,
    width,
    has_bias,
    activation=None,
    out=None,
    strip=16,
    vec=16,
    bs=256,
    hoist=False,
    probe=False,
    macro=1,
):
    """Compute forward causal 1D convolution using token strips.

    Applies causal convolution to channel-last inputs using token strips per thread. Requires
    sequence length greater than or equal to strip length. When the sequence length is not a
    multiple of strip length, the final strip shifts to align with the sequence boundary. When all
    loads are issued before any store, reads all input data tiles before issuing stores.

    Args:
        x: Input tensor of shape (batch, dim, seqlen) viewing contiguous (batch, seqlen, dim)
            storage.
        weight_tm: Time-major packed weight tensor of shape (width + int(has_bias), dim).
        width: Convolution kernel width.
        has_bias: Whether bias values are present in weight_tm.
        activation: Activation function to apply, either 'silu' or None.
        out: Optional output tensor of shape (batch, dim, seqlen) viewing contiguous (batch, seqlen,
            dim) memory. If None, a new tensor is allocated.
        strip: Number of tokens per strip processed by each thread. Default: 16.
        vec: Channel vector width per thread tile. Default: 16.
        bs: Thread block size. Default: 256.
        hoist: Whether all loads of a strip are issued before any store. Default: False.
        probe: Whether to bypass convolution arithmetic and copy input directly to output. Default:
            False.
        macro: Number of consecutive strips that one thread walks while keeping the last
            width - 1 tokens in registers. Requires seqlen >= strip * macro. Default: 1.

    Returns:
        Output tensor of shape (batch, dim, seqlen).
    """
    B, D, L = x.shape
    xs = x.transpose(1, 2)
    xstride = None if xs.is_contiguous() else channellast_row_stride(x, vec)
    span = strip * macro
    assert (xs.is_contiguous() or xstride) and L >= span
    if out is None:
        out = torch.empty((B, L, D), dtype=x.dtype, device=x.device).transpose(1, 2)
    xflat = (
        xs.reshape(1, -1)
        if xstride is None
        else x.as_strided((1, (B * L - 1) * xstride + D), ((B * L - 1) * xstride + D, 1))
    )
    nstrips = (L + span - 1) // span
    threads = nstrips * (D // vec)
    k = _compile_strip(
        x.dtype,
        weight_tm.dtype,
        D,
        width,
        has_bias,
        activation is not None,
        strip,
        vec,
        bs,
        hoist,
        probe,
        threads % bs == 0,
        L % span == 0,
        xstride,
        macro,
    )
    k(
        xflat,
        weight_tm,
        out.transpose(1, 2).reshape(1, -1),
        L,
        nstrips,
        (threads + bs - 1) // bs,
        B,
    )
    return out


class ConvFwdRowMajorRagged:
    """Execute row-major forward causal conv1d for arbitrary lengths.

    Row-major forward causal convolution for contiguous inputs of shape (batch, dim, seqlen) where
    seqlen is not required to divide by the vector width. Treats memory as a continuous 1D array
    while maintaining aligned vector loads. Masking prevents values from preceding sequences from
    leaking into the output.

    Supports three compile-time modes:
    - 'full': Evaluates sequence transitions and causal masking for all tiles.
    - 'interior': Evaluates convolution without boundary checks for tiles that do not cross sequence
      boundaries.
    - 'correction pass': Evaluates boundary arithmetic on sequence boundary tiles. Launched after
      'interior' on the same stream to overwrite boundary tiles.
    """

    def __init__(self, dim, width, has_bias, silu, vec=16, bs=256, mode="full"):
        """Initialize ragged row-major forward convolution parameters.

        Args:
            dim: Number of channels.
            width: Convolution kernel width.
            has_bias: Whether packed weights contain channel bias.
            silu: Whether to apply SiLU activation.
            vec: Vector width per thread tile.
            bs: Thread block size.
            mode: Execution mode, either 'full', 'interior', or 'correction pass'.
        """
        assert width - 1 <= vec and mode in ("full", "interior", "boundary")
        self.dim, self.width, self.has_bias, self.silu, self.vec, self.bs = (
            dim,
            width,
            has_bias,
            silu,
            vec,
            bs,
        )
        self.P, self.mode = (packed_width(width, has_bias), mode)

    @cute.jit
    def _load(self, tile):
        """Load a tile from global memory into registers.

        Args:
            tile: Tensor tile to load.

        Returns:
            Register tensor containing loaded values.
        """
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(
        self,
        x,
        wp,
        out,
        seqlen: Int32,
        nrows: Int32,
        ntiles: Int32,
        nthreads: Int32,
        nblocks: Int32,
        stream,
    ):
        """Launch the ragged row-major forward kernel over a 1D grid.

        Args:
            x: Input tensor in global memory.
            wp: Packed filter weights tensor in global memory.
            out: Output tensor in global memory.
            seqlen: Sequence length of each channel row.
            nrows: Total number of channel rows across the batch.
            ntiles: Total number of vector tiles.
            nthreads: Total number of threads to launch.
            nblocks: Number of thread blocks in the grid.
            stream: CUDA stream for execution.
        """
        self.kernel(x, wp, out, seqlen, nrows, ntiles, nthreads).launch(
            grid=[nblocks, 1, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, wp, out, seqlen: Int32, nrows: Int32, ntiles: Int32, nthreads: Int32):
        """Execute ragged row-major forward convolution for a single thread.

        Computes a vector tile across continuous memory. Evaluates row transitions, loads weights
        for current and subsequent rows, masks causal historical inputs, accumulates products in
        float32 precision, and writes output tiles. Out-of-range threads are mapped onto the last
        element.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        W: cutlass.Constexpr = self.width
        V: cutlass.Constexpr = self.vec
        P: cutlass.Constexpr = self.P
        q = bidx * self.bs + tidx
        q = q if q < nthreads else nthreads - 1
        i = q
        if const_expr(self.mode == "boundary"):
            i = q // 2 * seqlen // V + q % 2
            i = i if i < ntiles else ntiles - 1
        p0 = i * V
        r0 = p0 // seqlen
        t0 = p0 - r0 * seqlen
        xt = cute.logical_divide(x, (None, V))
        ip = i - 1 if i > 0 else i
        cur = self._load(xt[0, (None, i)])
        prev = self._load(xt[0, (None, ip)])
        wpt = cute.logical_divide(wp, (None, P))
        wa = self._load(wpt[r0 % self.dim, (None, 0)])
        zero = cutlass.Float32(0.0)
        waf = [wa[k].to(Float32) for k in range(P)]
        xs = [prev[V - (W - 1) + j].to(Float32) for j in range(W - 1)] + [
            cur[e].to(Float32) for e in range(V)
        ]
        orr = cute.make_rmem_tensor_like(cur)
        if const_expr(self.mode == "interior"):
            for e in cutlass.range_constexpr(V):
                a = waf[P - 1] if const_expr(self.has_bias) else zero
                for k in cutlass.range_constexpr(W):
                    a = a + xs[e + k] * waf[k]
                if const_expr(self.silu):
                    a = silu_f32(a, out.element_type.width == 32)
                orr[e] = a.to(out.element_type)
        else:
            r1 = r0 + 1 if r0 + 1 < nrows else r0
            wb = self._load(wpt[r1 % self.dim, (None, 0)])
            wbf = [wb[k].to(Float32) for k in range(P)]
            for e in cutlass.range_constexpr(V):
                te = t0 + e
                second = te >= seqlen
                t = te - seqlen if second else te
                a = (wbf[P - 1] if second else waf[P - 1]) if const_expr(self.has_bias) else zero
                for k in cutlass.range_constexpr(W):
                    lag = W - 1 - k
                    wk = wbf[k] if second else waf[k]
                    xv = xs[e + k]
                    if const_expr(lag > 0):
                        xv = xv if t >= lag else zero
                    a = a + xv * wk
                if const_expr(self.silu):
                    a = silu_f32(a, out.element_type.width == 32)
                orr[e] = a.to(out.element_type)
        cute.autovec_copy(orr, cute.logical_divide(out, (None, V))[0, (None, i)])


@functools.cache
def _compile_ragged(dtype, wdtype, dim, width, has_bias, silu, vec, bs, mode="full"):
    """Compile and cache the ragged row-major forward convolution kernel.

    Args:
        dtype: Data type of input and output tensors.
        wdtype: Data type of packed filter weights.
        dim: Number of channels.
        width: Convolution kernel width.
        has_bias: Whether packed weights include bias.
        silu: Whether to apply SiLU activation.
        vec: Vector width per thread tile.
        bs: Thread block size.
        mode: Kernel mode, either 'full', 'interior', or 'correction pass'.

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    P = packed_width(width, has_bias)
    n = cute.sym_int(divisibility=vec)
    flat = lambda: cute.runtime.make_fake_tensor(
        dt, (1, n), stride=(cute.sym_int64(divisibility=vec), 1), assumed_align=vec * dt.width // 8
    )
    wfake = cute.runtime.make_fake_tensor(
        wt, (dim, P), stride=(P, 1), assumed_align=P * wt.width // 8
    )
    return cute.compile(
        ConvFwdRowMajorRagged(dim, width, has_bias, silu, vec, bs, mode),
        flat(),
        wfake,
        flat(),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_fwd_rowmajor_ragged(
    x,
    weight_packed,
    width,
    has_bias,
    activation=None,
    out=None,
    vec=16,
    bs=256,
    two_pass=False,
    fix_bs=64,
):
    """Compute forward causal 1D convolution on unaligned row-major inputs.

    Applies causal convolution to contiguous row-major inputs with sequence length not required to
    divide by the vector width.

    Args:
        x: Contiguous input tensor of shape (batch, dim, seqlen) with seqlen >= vec and total
            elements divisible by vec.
        weight_packed: Packed filter weights of shape (dim, packed_width) produced by pack_weight.
        width: Convolution kernel width.
        has_bias: Whether bias values are present in weight_packed.
        activation: Activation function to apply, either 'silu' or None.
        out: Optional output tensor of shape (batch, dim, seqlen). If None, a new tensor is
            allocated.
        vec: Vector width per thread tile. Default: 16.
        bs: Thread block size for the primary pass. Default: 256.
        two_pass: Whether to run an interior pass followed by a boundary correction pass. Default:
            False.
        fix_bs: Thread block size for the boundary correction pass when two_pass is True. Default:
            64.

    Returns:
        Output tensor of shape (batch, dim, seqlen).
    """
    B, D, L = x.shape
    n = B * D * L
    assert x.is_contiguous() and L >= vec and (n % vec == 0)
    if out is None:
        out = torch.empty_like(x)
    tiles, rows, silu = (n // vec, B * D, activation is not None)
    xf, of = (x.view(1, -1), out.view(1, -1))
    if not two_pass:
        k = _compile_ragged(x.dtype, weight_packed.dtype, D, width, has_bias, silu, vec, bs, "full")
        k(xf, weight_packed, of, L, rows, tiles, tiles, (tiles + bs - 1) // bs)
        return out
    k = _compile_ragged(x.dtype, weight_packed.dtype, D, width, has_bias, silu, vec, bs, "interior")
    k(xf, weight_packed, of, L, rows, tiles, tiles, (tiles + bs - 1) // bs)
    kf = _compile_ragged(
        x.dtype, weight_packed.dtype, D, width, has_bias, silu, vec, fix_bs, "boundary"
    )
    kf(xf, weight_packed, of, L, rows, tiles, 2 * rows, (2 * rows + fix_bs - 1) // fix_bs)
    return out
