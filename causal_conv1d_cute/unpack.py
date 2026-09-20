"""Single-token decode that also unpacks a fused projection.

Gated DeltaNet layers project the hidden state once into a row [Q | K | V | Z] and once into a row
[B | A]. The conv runs over the leading Q, K, V columns; Z, B and A go to later kernels that want
them as separate contiguous tensors. A serving engine that decodes one token per sequence therefore
needs, per layer, a conv state update plus three small copies. This kernel does all four in one
launch, reading the conv input in place from the projection row.

The kernel reads the filter weights and the bias from the model's own (dim, width) and (dim,)
tensors. It keeps no packed copy: inside a serving engine even a few megabytes allocated by a plugin
move every later buffer of the server, and with one request in flight that alone changed the decode
step time by up to 0.3% in either direction, more than the kernel saves.

Each thread owns one tile of vec adjacent conv channels of one batch row. It loads the filter
weights, the conv state, the conv input, and one tile each of Z, B and A, then stores. Threads whose
index lies beyond the last Z, B or A tile load the last one again and do not store it. Every load is
issued before the first store, the copies are stored before the convolution is computed, and the
convolution is the last block of the kernel, which the fused bfloat16 multiply-add requires.

Batch rows whose slot is the pad slot or lies outside the state cache keep their conv state, and
their conv input is copied to the output unchanged.
"""

import functools
import math
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from .fwd import _DTYPES, silu_f32


class ConvUpdateUnpack:
    """Indexed single-token decode over the leading columns of a [Q | K | V | Z] row."""

    def __init__(self, dim, zdim, heads, width, has_bias, silu, vec=16, bs=32):
        """Initialize the kernel parameters.

        Args:
            dim: Number of conv channels, the width of Q, K and V together.
            zdim: Width of Z.
            heads: Width of B and of A.
            width: Kernel width.
            has_bias: Whether a bias is added.
            silu: Whether to apply SiLU activation to the conv outputs.
            vec: Number of adjacent channels per tile.
            bs: Thread block size along the tile dimension.
        """
        self.hv = math.gcd(heads, vec)
        assert dim % vec == 0 and zdim % vec == 0 and dim // vec % bs == 0
        assert zdim <= dim and heads // self.hv <= dim // vec
        self.dim, self.zdim, self.heads, self.width = (dim, zdim, heads, width)
        self.has_bias, self.silu, self.vec, self.bs = (has_bias, silu, vec, bs)
        self.K = width - 1
        self.wv = math.gcd(vec * width, 16)
        self.G, self.GZ, self.GH = (dim // vec, zdim // vec, heads // self.hv)

    @cute.jit
    def _load(self, tile):
        """Load a tensor tile into registers.

        Args:
            tile: Tensor tile to load.

        Returns:
            Register tensor containing the loaded tile.
        """
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(
        self, qkvz, ba, state, w, bias, out, z, b, a, idx, pad: Int32, slots: Int32, nb, stream
    ):
        """Launch the kernel on a CUDA stream.

        Args:
            qkvz: Projection rows of shape (batch, dim + zdim).
            ba: Projection rows of shape (batch, 2 * heads).
            state: Conv state cache of shape (slots, dim * (width - 1)), updated in place.
            w: Filter weights of shape (1, dim * width), channel by channel.
            bias: Bias of shape (1, dim). Ignored unless has_bias.
            out: Conv outputs of shape (batch, dim).
            z: Copy of the Z columns, shape (batch, zdim).
            b: Copy of the B columns, shape (batch, heads).
            a: Copy of the A columns, shape (batch, heads).
            idx: Conv state slot of each batch row, shape (batch,).
            pad: Slot index marking batch rows to skip.
            slots: Number of slots in the conv state cache.
            nb: Number of batch rows.
            stream: CUDA stream for launch.
        """
        self.kernel(qkvz, ba, state, w, bias, out, z, b, a, idx, pad, slots).launch(
            grid=[self.G // self.bs, nb, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.jit
    def _store(self, reg, tile):
        """Store a register tensor to a tensor tile.

        Args:
            reg: Register tensor.
            tile: Destination tile.
        """
        cute.autovec_copy(reg, tile)

    @cute.jit
    def _store2(self, reg0, tile0, reg1, tile1):
        """Store two register tensors to two tensor tiles.

        Args:
            reg0: First register tensor.
            tile0: Destination of the first register tensor.
            reg1: Second register tensor.
            tile1: Destination of the second register tensor.
        """
        cute.autovec_copy(reg0, tile0)
        cute.autovec_copy(reg1, tile1)

    @cute.jit
    def _conv(self, out, state, wr, br, sr, xr, slot, bi, j):
        """Compute the conv outputs of one tile and shift its conv state.

        Args:
            out: Conv outputs, one row per batch row.
            state: Conv state cache, one row per slot.
            wr: Register tensors holding the filter weights of the tile, channel by channel.
            br: Register tensor holding the bias of the tile, or None.
            sr: Register tensors holding the conv state of the tile.
            xr: Register tensor holding the conv input of the tile.
            slot: Conv state slot of this batch row.
            bi: Batch row.
            j: Tile index along the channel dimension.
        """
        W: cutlass.Constexpr = self.width
        K: cutlass.Constexpr = self.K
        V: cutlass.Constexpr = self.vec
        WV: cutlass.Constexpr = self.wv
        ot = cute.logical_divide(out, (None, V))
        st = cute.logical_divide(state, (None, V))
        orr = cute.make_rmem_tensor_like(xr)
        ns = [cute.make_rmem_tensor_like(xr) for _ in range(K)]
        for e in cutlass.range_constexpr(V):
            a = br[e].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
            for k in cutlass.range_constexpr(K):
                sv = sr[(e * K + k) // V][(e * K + k) % V]
                a = a + sv.to(Float32) * wr[(e * W + k) // WV][(e * W + k) % WV].to(Float32)
                if const_expr(k > 0):
                    ns[(e * K + k - 1) // V][(e * K + k - 1) % V] = sv
            ns[(e * K + K - 1) // V][(e * K + K - 1) % V] = xr[e]
            a = a + xr[e].to(Float32) * wr[(e * W + K) // WV][(e * W + K) % WV].to(Float32)
            if const_expr(self.silu):
                a = silu_f32(a, out.element_type.width == 32)
            orr[e] = a.to(out.element_type)
        for k in cutlass.range_constexpr(K):
            cute.autovec_copy(ns[k], st[slot, (None, j * K + k)])
        cute.autovec_copy(orr, ot[bi, (None, j)])

    @cute.kernel
    def kernel(self, qkvz, ba, state, w, bias, out, z, b, a, idx, pad: Int32, slots: Int32):
        """Load everything a tile needs, store the copies, then run the convolution.

        Args:
            qkvz: Projection rows of shape (batch, dim + zdim).
            ba: Projection rows of shape (batch, 2 * heads).
            state: Conv state cache, one row per slot.
            w: Filter weights, channel by channel.
            bias: Bias.
            out: Conv outputs of shape (batch, dim).
            z: Copy of the Z columns.
            b: Copy of the B columns.
            a: Copy of the A columns.
            idx: Conv state slot of each batch row.
            pad: Slot index marking batch rows to skip.
            slots: Number of slots in the conv state cache.
        """
        W: cutlass.Constexpr = self.width
        K: cutlass.Constexpr = self.K
        V: cutlass.Constexpr = self.vec
        HV: cutlass.Constexpr = self.hv
        GZ: cutlass.Constexpr = self.GZ
        GH: cutlass.Constexpr = self.GH
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bi, _ = cute.arch.block_idx()
        j = bidx * self.bs + tidx
        slot = idx[bi]
        valid = (slot != pad) & (slot >= 0) & (slot < slots)
        skip = (slot == pad) | (slot < 0) | (slot >= slots)
        ls = slot if valid else 0
        jz = j if j < GZ else GZ - 1
        jh = j if j < GH else GH - 1
        xt = cute.logical_divide(qkvz, (None, V))
        st = cute.logical_divide(state, (None, V))
        WV: cutlass.Constexpr = self.wv
        NW: cutlass.Constexpr = self.vec * self.width // self.wv
        wt = cute.logical_divide(w, (None, WV))
        gt = cute.logical_divide(ba, (None, HV))
        wr = [self._load(wt[0, (None, j * NW + m)]) for m in range(NW)]
        br = (
            self._load(cute.logical_divide(bias, (None, V))[0, (None, j)])
            if const_expr(self.has_bias)
            else None
        )
        sr = [self._load(st[ls, (None, j * K + k)]) for k in range(K)]
        xr = self._load(xt[bi, (None, j)])
        zr = self._load(xt[bi, (None, self.G + jz)])
        b_r = self._load(gt[bi, (None, jh)])
        a_r = self._load(gt[bi, (None, GH + jh)])
        if j < GZ:
            self._store(zr, cute.logical_divide(z, (None, V))[bi, (None, j)])
        if j < GH:
            self._store2(
                b_r,
                cute.logical_divide(b, (None, HV))[bi, (None, j)],
                a_r,
                cute.logical_divide(a, (None, HV))[bi, (None, j)],
            )
        if skip:
            self._store(xr, cute.logical_divide(out, (None, V))[bi, (None, j)])
        if valid:
            self._conv(out, state, wr, br, sr, xr, slot, bi, j)


def _rows(dtype, cols, vec):
    """Create a fake two-dimensional tensor with a symbolic row count and row stride.

    Args:
        dtype: Element data type.
        cols: Number of columns.
        vec: Number of elements that the row stride is a multiple of and that rows are aligned to.

    Returns:
        Fake tensor descriptor for compilation.
    """
    return cute.runtime.make_fake_tensor(
        dtype,
        (cute.sym_int(), cols),
        stride=(cute.sym_int64(divisibility=vec), 1),
        assumed_align=vec * dtype.width // 8,
    )


@functools.cache
def _compile_unpack(dtype, gdtype, wdtype, dim, zdim, heads, width, has_bias, silu, vec, bs):
    """Compile the decode and unpack kernel for given configurations.

    Args:
        dtype: Data type of the [Q | K | V | Z] rows and of the conv state.
        gdtype: Data type of the [B | A] rows.
        wdtype: Weight data type.
        dim: Number of conv channels.
        zdim: Width of Z.
        heads: Width of B and of A.
        width: Kernel width.
        has_bias: Whether a bias is added.
        silu: Whether SiLU activation is enabled.
        vec: Number of adjacent channels per tile.
        bs: Thread block size along the tile dimension.

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, gt, wt = (_DTYPES[dtype], _DTYPES[gdtype], _DTYPES[wdtype])
    hv = math.gcd(heads, vec)
    ints = cute.runtime.make_fake_tensor(
        cutlass.Int32, (cute.sym_int(),), stride=(1,), assumed_align=4
    )
    return cute.compile(
        ConvUpdateUnpack(dim, zdim, heads, width, has_bias, silu, vec, bs),
        _rows(dt, dim + zdim, vec),
        _rows(gt, 2 * heads, hv),
        _rows(dt, dim * (width - 1), vec),
        _rows(wt, dim * width, math.gcd(vec * width, 16)),
        _rows(wt, dim, vec),
        _rows(dt, dim, vec),
        _rows(dt, zdim, vec),
        _rows(gt, heads, hv),
        _rows(gt, heads, hv),
        ints,
        Int32(1),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def unpack_reason(qkvz, ba, conv_state, weight, bias, heads):
    """Return why a decode and unpack call cannot be served, or None when it can.

    Args:
        qkvz: Projection rows of shape (batch, dim + zdim).
        ba: Projection rows of shape (batch, 2 * heads).
        conv_state: Conv state cache of shape (slots, dim, width - 1).
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        heads: Width of B and of A.

    Returns:
        A short description of the first unmet requirement, or None.
    """
    D, W = weight.shape
    zdim = qkvz.shape[1] - D
    if qkvz.dtype not in _DTYPES or ba.dtype not in _DTYPES or weight.dtype != qkvz.dtype:
        return f"dtypes {qkvz.dtype}, {ba.dtype}, {weight.dtype}"
    if not 2 <= W <= 8 or D % 16 != 0 or zdim <= 0 or zdim % 16 != 0 or zdim > D:
        return f"conv of {D} channels and width {W} with {zdim} further columns"
    vec = math.gcd(16, qkvz.stride(0))
    if qkvz.stride(1) != 1 or ba.stride(1) != 1 or vec < 4 or ba.shape[1] != 2 * heads:
        return f"projection strides {qkvz.stride()}, {ba.stride()}"
    if ba.stride(0) % math.gcd(heads, vec) or math.gcd(heads, vec) < 2:
        return f"{heads} heads with row stride {ba.stride(0)}"
    if conv_state.dtype != qkvz.dtype or tuple(conv_state.shape[1:]) != (D, W - 1):
        return f"conv state {tuple(conv_state.shape)} {conv_state.dtype}"
    if not conv_state.is_contiguous() or not weight.is_contiguous():
        return "conv state or weight is not contiguous"
    if bias is not None and (bias.dtype != weight.dtype or bias.stride(0) != 1):
        return "bias dtype or stride"
    align = 16 * qkvz.element_size()
    ptrs = [qkvz.data_ptr(), conv_state.data_ptr(), weight.data_ptr()]
    ptrs += [] if bias is None else [bias.data_ptr()]
    if any(ptr % align for ptr in ptrs) or ba.data_ptr() % (16 * ba.element_size()):
        return "tensors are not aligned to 16 elements"
    return None


def causal_conv1d_update_unpack(
    qkvz, ba, conv_state, weight, bias, conv_state_indices, heads, activation, pad, vec, bs
):
    """Advance an indexed conv state by one token and unpack Z, B and A.

    Args:
        qkvz: Projection rows of shape (batch, dim + zdim); the conv reads the leading dim columns.
        ba: Projection rows of shape (batch, 2 * heads).
        conv_state: Conv state cache of shape (slots, dim, width - 1), updated in place.
        weight: Filter weights of shape (dim, width), contiguous. Read in place, not copied.
        bias: Optional bias tensor of shape (dim,).
        conv_state_indices: Int32 tensor of shape (batch,) mapping batch rows to state slots.
        heads: Width of B and of A.
        activation: Activation function to apply, either "silu" or None.
        pad: Slot index marking batch rows to skip.
        vec: Number of adjacent channels per tile.
        bs: Thread block size along the tile dimension.

    Returns:
        Tuple of the conv outputs of shape (batch, dim), and contiguous copies of Z of shape
        (batch, zdim), B of shape (batch, heads) and A of shape (batch, heads).
    """
    B, (D, width) = (qkvz.shape[0], weight.shape)
    zdim = qkvz.shape[1] - D
    out = torch.empty((B, D), dtype=qkvz.dtype, device=qkvz.device)
    z = torch.empty((B, zdim), dtype=qkvz.dtype, device=qkvz.device)
    b = torch.empty((B, heads), dtype=ba.dtype, device=ba.device)
    a = torch.empty_like(b)
    k = _compile_unpack(
        qkvz.dtype,
        ba.dtype,
        weight.dtype,
        D,
        zdim,
        heads,
        width,
        bias is not None,
        activation is not None,
        vec,
        bs,
    )
    slots = conv_state.shape[0]
    wflat = weight.view(1, -1)
    bflat = (bias if bias is not None else wflat[0, :D]).view(1, D)
    state = conv_state.view(slots, -1)
    k(qkvz, ba, state, wflat, bflat, out, z, b, a, conv_state_indices, pad, slots, B)
    return (out, z, b, a)


def unpack_config(dim, batch, stride):
    """Pick the tile width and thread block size of the decode and unpack kernel.

    Tiles of 4 channels in blocks of 64 threads up to 32 batch rows, where latency is the latency of
    one thread, and tiles of 16 in blocks of 32 beyond, where fewer memory transactions per channel
    matter more. Measured alone for 1 to 256 rows and confirmed inside a server at 16 and 64
    concurrent requests, where one fixed choice of 16 and 64 cost 0.3 us and 0.7 us more per call.
    Every projection row has to start on a tile boundary, so the tile width is reduced to a divisor
    of the row stride; a tensor-parallel shard whose [Q | K | V | Z] and [B | A] projections share
    one buffer has rows that are a multiple of 8 but not of 16 elements apart. The block size is
    halved until it divides the tile count, so that the launch covers exactly the work.

    Args:
        dim: Number of conv channels.
        batch: Number of batch rows.
        stride: Distance in elements between consecutive projection rows.

    Returns:
        Tuple of channels per tile and thread block size.
    """
    vec, bs = (4, 64) if batch <= 32 else (16, 32)
    vec = math.gcd(vec, stride)
    while dim // vec % bs:
        bs //= 2
    return (vec, bs)
