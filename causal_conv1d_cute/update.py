"""Update causal 1D conv state and compute single-token decode outputs.

Computes out[b, c] = act(bias[c] + sum_k weight[c, k] * window[b, c, k]) where window =
[conv_state[b, c, :], x[b, c]], and updates conv state in place. Accumulation is performed in fp32.
Supports kernel width from 2 to 8.
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
    """Compute SiLU activation in fp32 using hardware tanh.

    Evaluates x * sigmoid(x) via 0.5 * x * (tanh(0.5 * x) + 1.0). When exact is True, uses standard
    tanh for fp32 outputs. When exact is False, uses approximate tanh for 16-bit outputs.

    Args:
        a: Input value in fp32.
        exact: Whether to evaluate tanh with standard precision.

    Returns:
        Activated value in fp32.
    """
    half = Float32(0.5)
    if const_expr(exact):
        return a * (cute.math.tanh(a * half) * half + half)
    return a * (cute.math.tanh(a * half, approx=True) * half + half)


class ConvUpdateScalar:
    """Execute single-token conv update using one thread per channel.

    Reference implementation using one thread per (batch, channel) with scalar loads and stores.
    Accumulates in fp32 and updates conv state in place.
    """

    def __init__(self, width: int, has_bias: bool, silu: bool, bs: int = 128):
        """Initialize scalar decode kernel parameters."""
        self.width, self.has_bias, self.silu, self.bs = (width, has_bias, silu, bs)

    @cute.jit
    def __call__(self, x, state, w, b, out, n: Int32, stream):
        """Launch scalar decode kernel over n elements."""
        self.kernel(x, state, w, b, out, n).launch(
            grid=[(n + self.bs - 1) // self.bs, 1, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, state, w, b, out, n: Int32):
        """Compute single-token conv update with one thread per channel.

        Accumulates in fp32, updates conv state in place, applies optional bias and SiLU activation,
        and writes output.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        W: cutlass.Constexpr = self.width
        i = bidx * self.bs + tidx
        dim = cute.size(x, mode=[1])
        if i < n:
            bi = i // dim
            c = i % dim
            acc = Float32(0.0)
            if const_expr(self.has_bias):
                acc = b[c].to(Float32)
            for k in cutlass.range_constexpr(W - 1):
                s = state[bi, c, k]
                acc = acc + s.to(Float32) * w[c, k].to(Float32)
                if const_expr(k > 0):
                    state[bi, c, k - 1] = s
            xv = x[bi, c]
            state[bi, c, W - 2] = xv
            acc = acc + xv.to(Float32) * w[c, W - 1].to(Float32)
            if const_expr(self.silu):
                acc = silu_f32(acc, out.element_type.width == 32)
            out[bi, c] = acc.to(out.element_type)


class ConvUpdateVec:
    """Execute vectorized channel-major single-token decode kernel.

    Assigns one thread per cv channels. Operates on channel-major conv state shaped (batch, dim *
    (kernel width - 1)). Issues vector loads for conv state, filter weights, input, and bias.
    Accumulates element-wise in registers in fp32 across cv independent chains and updates conv
    state in place.

    When the thread count equals the work count ((dim // cv) % bs == 0), boundary checks are
    omitted.
    """

    def __init__(
        self,
        dim: int,
        width: int,
        has_bias: bool,
        silu: bool,
        cv: int = 16,
        bs: int = 32,
        tiles: int = 1,
    ):
        """Initialize vectorized channel-major decode kernel parameters.

        Args:
            dim: Number of channels.
            width: Kernel width.
            has_bias: Whether a bias is added.
            silu: Whether to apply SiLU activation.
            cv: Number of adjacent channels per tile.
            bs: Thread block size.
            tiles: Number of consecutive tiles that one thread handles, one after the other.
        """
        assert dim % cv == 0 and dim * (width - 1) % cv == 0 and (dim * width % cv == 0)
        assert dim // cv % tiles == 0
        self.dim, self.width, self.has_bias, self.silu, self.cv, self.bs = (
            dim,
            width,
            has_bias,
            silu,
            cv,
            bs,
        )
        self.groups = dim // cv
        self.st = math.gcd(cv * (width - 1), 16)
        self.wt = math.gcd(cv * width, 16)
        self.T = tiles
        self.gt = self.groups // tiles
        self.guard = self.gt % bs != 0

    @cute.jit
    def _load(self, tile):
        """Load a tile into register memory using autovec_copy."""
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(self, x, state, w, b, out, n: Int32, stream):
        """Launch vectorized decode kernel over n channel groups."""
        self.kernel(x, state, w, b, out, n).launch(
            grid=[(n + self.bs - 1) // self.bs, 1, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.jit
    def _body(self, x, state, w, b, out, bi, g):
        """Execute vectorized decode for channel group i.

        All loads are issued before any store. Forms cv independent fp32 accumulation chains in
        registers, updates conv state in place, and stores output vectors.
        """
        W: cutlass.Constexpr = self.width
        CV: cutlass.Constexpr = self.cv
        xt = cute.logical_divide(x, (None, CV))
        ot = cute.logical_divide(out, (None, CV))
        ST: cutlass.Constexpr = self.st
        WT: cutlass.Constexpr = self.wt
        NS: cutlass.Constexpr = CV * (W - 1) // self.st
        NW: cutlass.Constexpr = CV * W // self.wt
        st = cute.logical_divide(state, (None, ST))
        wt = cute.logical_divide(w, (None, WT))
        xr = self._load(xt[bi, (None, g)])
        sr = [self._load(st[bi, (None, g * NS + k)]) for k in range(NS)]
        wr = [self._load(wt[0, (None, g * NW + k)]) for k in range(NW)]
        br = (
            self._load(cute.logical_divide(b, (None, CV))[0, (None, g)])
            if const_expr(self.has_bias)
            else None
        )
        ns = [cute.make_rmem_tensor_like(sr[0]) for _ in range(NS)]
        orr = cute.make_rmem_tensor_like(xr)
        for c in cutlass.range_constexpr(CV):
            a = br[c].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
            for k in cutlass.range_constexpr(W - 1):
                sv = sr[(c * (W - 1) + k) // ST][(c * (W - 1) + k) % ST]
                a = a + sv.to(Float32) * wr[(c * W + k) // WT][(c * W + k) % WT].to(Float32)
                if const_expr(k > 0):
                    dst_vec, dst_elt = divmod(c * (W - 1) + k - 1, ST)
                    ns[dst_vec][dst_elt] = sv
            xv = xr[c]
            last_vec, last_elt = divmod(c * (W - 1) + W - 2, ST)
            ns[last_vec][last_elt] = xv
            a = a + xv.to(Float32) * wr[(c * W + W - 1) // WT][(c * W + W - 1) % WT].to(Float32)
            if const_expr(self.silu):
                a = silu_f32(a, out.element_type.width == 32)
            orr[c] = a.to(out.element_type)
        for k in cutlass.range_constexpr(NS):
            cute.autovec_copy(ns[k], st[bi, (None, g * NS + k)])
        cute.autovec_copy(orr, ot[bi, (None, g)])

    @cute.kernel
    def kernel(self, x, state, w, b, out, n: Int32):
        """Execute vectorized channel-major decode kernel on grid.

        Applies boundary checks when the thread count does not equal the work count.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * self.bs + tidx
        if const_expr(self.guard):
            if i < n:
                self._run(x, state, w, b, out, i)
        else:
            self._run(x, state, w, b, out, i)

    @cute.jit
    def _run(self, x, state, w, b, out, i):
        """Handle the tiles of thread i one after the other.

        Args:
            x: Input activations of shape (batch, dim).
            state: Conv state of shape (batch, dim * (width - 1)).
            w: Filter weights of shape (1, dim * width).
            b: Bias of shape (1, dim).
            out: Output activations of shape (batch, dim).
            i: Thread index, below batch * (dim // cv // tiles).
        """
        bi = i // self.gt
        g0 = (i - bi * self.gt) * self.T
        for q in cutlass.range_constexpr(self.T):
            self._body(x, state, w, b, out, bi, g0 + q)


class ConvUpdateTimeMajor:
    """Execute time-major single-token decode kernel.

    Operates on conv state physically stored in (batch, kernel width - 1, dim) order and transposed
    filter weights of shape (kernel width, dim). Each chunk of cv channels uses aligned vector loads
    of conv state and filter weights. Updates conv state in place by storing row k + 1 into row k.
    Accumulation is performed in fp32, either across vector operations or per-element scalar chains.
    """

    def __init__(
        self,
        dim: int,
        width: int,
        has_bias: bool,
        silu: bool,
        cv: int = 16,
        bs: int = 32,
        vec_acc: bool = False,
    ):
        """Initialize time-major decode kernel parameters."""
        assert dim % cv == 0
        self.dim, self.width, self.has_bias, self.silu, self.cv, self.bs = (
            dim,
            width,
            has_bias,
            silu,
            cv,
            bs,
        )
        self.groups = dim // cv
        self.guard = self.groups % bs != 0
        self.vec_acc = vec_acc

    @cute.jit
    def _load(self, tile):
        """Load a tile into register memory using autovec_copy."""
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(self, x, state, wt, b, out, n: Int32, stream):
        """Launch time-major decode kernel over n channel groups."""
        self.kernel(x, state, wt, b, out, n).launch(
            grid=[(n + self.bs - 1) // self.bs, 1, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.jit
    def _body(self, x, state, wt, b, out, i):
        """Execute time-major decode for channel group i.

        Loads aligned conv state and filter weight vectors, accumulates in fp32, updates conv state
        rows in place, and stores output.
        """
        W: cutlass.Constexpr = self.width
        CV: cutlass.Constexpr = self.cv
        G: cutlass.Constexpr = self.groups
        bi = i // G
        g = i % G
        xt = cute.logical_divide(x, (None, CV))
        ot = cute.logical_divide(out, (None, CV))
        st = cute.logical_divide(state, (None, CV))
        wtt = cute.logical_divide(wt, (None, CV))
        xr = self._load(xt[bi, (None, g)])
        sr = [self._load(st[bi, (None, k * G + g)]) for k in range(W - 1)]
        wr = [self._load(wtt[0, (None, k * G + g)]) for k in range(W)]
        br = (
            self._load(cute.logical_divide(b, (None, CV))[0, (None, g)])
            if const_expr(self.has_bias)
            else None
        )
        orr = cute.make_rmem_tensor_like(xr)
        if const_expr(self.vec_acc):
            acc = xr.load().to(Float32) * wr[W - 1].load().to(Float32)
            if const_expr(self.has_bias):
                acc = acc + br.load().to(Float32)
            for k in cutlass.range_constexpr(W - 1):
                acc = acc + sr[k].load().to(Float32) * wr[k].load().to(Float32)
            if const_expr(self.silu):
                acc = silu_f32(acc, out.element_type.width == 32)
            orr.store(acc.to(out.element_type))
        else:
            for c in cutlass.range_constexpr(CV):
                a = br[c].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
                for k in cutlass.range_constexpr(W - 1):
                    a = a + sr[k][c].to(Float32) * wr[k][c].to(Float32)
                a = a + xr[c].to(Float32) * wr[W - 1][c].to(Float32)
                if const_expr(self.silu):
                    a = silu_f32(a, out.element_type.width == 32)
                orr[c] = a.to(out.element_type)
        for k in cutlass.range_constexpr(W - 2):
            cute.autovec_copy(sr[k + 1], st[bi, (None, k * G + g)])
        cute.autovec_copy(xr, st[bi, (None, (W - 2) * G + g)])
        cute.autovec_copy(orr, ot[bi, (None, g)])

    @cute.kernel
    def kernel(self, x, state, wt, b, out, n: Int32):
        """Execute time-major decode kernel on grid.

        Applies boundary checks when the thread count does not equal the work count.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * self.bs + tidx
        if const_expr(self.guard):
            if i < n:
                self._body(x, state, wt, b, out, i)
        else:
            self._body(x, state, wt, b, out, i)


class ConvUpdatePadded:
    """Execute power-of-two padded single-token decode kernel.

    Assigns one thread per channel. Conv state and filter weights are padded to P, the next power of
    2 greater than or equal to kernel width. Fetches entire channel state and filter weight rows
    using aligned vector loads. Updates conv state in place by shifting the P-element state left by
    one and appending the new input token. Uses a 2D grid over channel blocks and batch rows.
    """

    def __init__(
        self, dim: int, width: int, has_bias: bool, silu: bool, bs: int = 128, mbp: int = 0
    ):
        """Initialize padded decode kernel parameters."""
        self.dim, self.width, self.has_bias, self.silu, self.bs = (dim, width, has_bias, silu, bs)
        self.P = 1 << (width - 1).bit_length()
        self.guard = dim % bs != 0
        self.mbp = mbp

    @cute.jit
    def _load(self, tile):
        """Load a tile into register memory using autovec_copy."""
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(self, x, state, wp, b, out, nb: Int32, stream):
        """Launch padded decode kernel across channel blocks and batch rows."""
        if const_expr(self.mbp > 0):
            self.kernel(x, state, wp, b, out).launch(
                grid=[(self.dim + self.bs - 1) // self.bs, nb, 1],
                block=[self.bs, 1, 1],
                stream=stream,
                min_blocks_per_mp=self.mbp,
            )
        else:
            self.kernel(x, state, wp, b, out).launch(
                grid=[(self.dim + self.bs - 1) // self.bs, nb, 1],
                block=[self.bs, 1, 1],
                stream=stream,
            )

    @cute.jit
    def _body(self, x, state, wp, b, out, bi, c):
        """Execute padded decode for batch row bi and channel c.

        Loads conv state and filter weight vectors, accumulates over the trailing kernel width - 1
        elements plus input x in fp32, shifts conv state in place, and stores output.
        """
        W: cutlass.Constexpr = self.width
        P: cutlass.Constexpr = self.P
        st = cute.logical_divide(state, (None, P))
        sr = self._load(st[bi * self.dim + c, (None, 0)])
        wr = self._load(cute.logical_divide(wp, (None, P))[c, (None, 0)])
        xv = x[bi, c]
        a = b[c].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
        for k in cutlass.range_constexpr(W - 1):
            a = a + sr[P - (W - 1) + k].to(Float32) * wr[k].to(Float32)
        a = a + xv.to(Float32) * wr[W - 1].to(Float32)
        if const_expr(self.silu):
            a = silu_f32(a, out.element_type.width == 32)
        ns = cute.make_rmem_tensor_like(sr)
        for j in cutlass.range_constexpr(P - 1):
            ns[j] = sr[j + 1]
        ns[P - 1] = xv
        cute.autovec_copy(ns, st[bi * self.dim + c, (None, 0)])
        out[bi, c] = a.to(out.element_type)

    @cute.kernel
    def kernel(self, x, state, wp, b, out):
        """Execute padded decode over a 2D grid.

        Executes thread body over channel blocks and batch rows with boundary checks.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bi, _ = cute.arch.block_idx()
        c = bidx * self.bs + tidx
        if const_expr(self.guard):
            if c < self.dim:
                self._body(x, state, wp, b, out, bi, c)
        else:
            self._body(x, state, wp, b, out, bi, c)


class ConvUpdatePaddedVec:
    """Execute vectorized padded single-token decode kernel.

    Processes cv channels per thread using conv state and filter weights padded to power-of-two
    stride P. Updates conv state in place. Requires that the thread count equals the work count
    ((dim // cv) is a multiple of block size). Out-of-range threads cannot be mapped onto the last
    element because conv state is updated in place, which would cause data races.
    """

    def __init__(self, dim, width, has_bias, silu, cv=16, bs=32):
        """Initialize vectorized padded decode kernel parameters."""
        self.dim, self.width, self.has_bias, self.silu, self.cv, self.bs = (
            dim,
            width,
            has_bias,
            silu,
            cv,
            bs,
        )
        self.P = 1 << (width - 1).bit_length()
        assert dim % cv == 0
        self.groups = dim // cv
        self.tw = min(16, cv * self.P)
        self.nt = cv * self.P // self.tw

    @cute.jit
    def _load(self, tile):
        """Load a tile into register memory using autovec_copy."""
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(self, x, state, wp, b, out, nthreads: Int32, nblocks: Int32, stream):
        """Launch vectorized padded decode kernel."""
        self.kernel(x, state, wp, b, out, nthreads).launch(
            grid=[nblocks, 1, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, state, wp, b, out, nthreads: Int32):
        """Execute vectorized padded decode kernel on grid.

        Executes vectorized decode and in-place conv state updates assuming the thread count equals
        the work count.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        W: cutlass.Constexpr = self.width
        P: cutlass.Constexpr = self.P
        CV: cutlass.Constexpr = self.cv
        G: cutlass.Constexpr = self.groups
        i = bidx * self.bs + tidx
        bi = i // G
        g = i - bi * G
        TW: cutlass.Constexpr = self.tw
        NT: cutlass.Constexpr = self.nt
        st = cute.logical_divide(state, (None, TW))
        wt = cute.logical_divide(wp, (None, TW))
        xr = self._load(cute.logical_divide(x, (None, CV))[bi, (None, g)])
        sr = [self._load(st[bi, (None, g * NT + n)]) for n in range(NT)]
        wr = [self._load(wt[0, (None, g * NT + n)]) for n in range(NT)]
        br = (
            self._load(cute.logical_divide(b, (None, CV))[0, (None, g)])
            if const_expr(self.has_bias)
            else None
        )
        ns = [cute.make_rmem_tensor_like(sr[0]) for _ in range(NT)]
        orr = cute.make_rmem_tensor_like(xr)
        for c in cutlass.range_constexpr(CV):
            a = br[c].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
            for k in cutlass.range_constexpr(W - 1):
                f = c * P + (P - (W - 1)) + k
                a = a + sr[f // TW][f % TW].to(Float32) * wr[(c * P + k) // TW][
                    (c * P + k) % TW
                ].to(Float32)
            xv = xr[c]
            a = a + xv.to(Float32) * wr[(c * P + W - 1) // TW][(c * P + W - 1) % TW].to(Float32)
            if const_expr(self.silu):
                a = silu_f32(a, out.element_type.width == 32)
            orr[c] = a.to(out.element_type)
            for jj in cutlass.range_constexpr(P - 1):
                src, dst = (c * P + jj + 1, c * P + jj)
                dv, de = divmod(dst, TW)
                ns[dv][de] = sr[src // TW][src % TW]
            lv, le = divmod(c * P + P - 1, TW)
            ns[lv][le] = xv
        for n in cutlass.range_constexpr(NT):
            cute.autovec_copy(ns[n], st[bi, (None, g * NT + n)])
        cute.autovec_copy(orr, cute.logical_divide(out, (None, CV))[bi, (None, g)])


class ConvUpdateStd2D:
    """Execute single-token decode kernel using 2D grid launch.

    Operates on standard conv state of shape (batch, dim, kernel width - 1). Filter weights and
    optional bias are stored in the same row of shape (dim, P), where P is the next power of 2
    greater than or equal to kernel width plus bias, allowing all filter weights and bias to be
    fetched in a single vector load. Conv state entries are accessed via scalar loads and stores and
    updated in place. Uses a 2D grid over channel blocks and batch rows.
    """

    def __init__(self, dim, width, has_bias, silu, bs=256):
        """Initialize 2D decode kernel parameters."""
        self.dim, self.width, self.has_bias, self.silu, self.bs = (dim, width, has_bias, silu, bs)
        self.P = 1 << (width + int(has_bias) - 1).bit_length()
        self.guard = dim % bs != 0

    @cute.jit
    def _load(self, tile):
        """Load a tile into register memory using autovec_copy."""
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(self, x, state, wp, out, nb: Int32, stream):
        """Launch 2D decode kernel across channel blocks and batch rows."""
        self.kernel(x, state, wp, out).launch(
            grid=[(self.dim + self.bs - 1) // self.bs, nb, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.jit
    def _body(self, x, state, wp, out, bi, c):
        """Execute decode for batch row bi and channel c.

        Loads filter weights and bias in one vector load, performs scalar loads for conv state and
        input x, accumulates in fp32, updates conv state in place, and writes output.
        """
        W: cutlass.Constexpr = self.width
        P: cutlass.Constexpr = self.P
        _iket.push("loads")
        wr = self._load(cute.logical_divide(wp, (None, P))[c, (None, 0)])
        sv = [state[bi, c, k] for k in range(W - 1)]
        xv = x[bi, c]
        _iket.pop()
        _iket.push("compute")
        a = wr[P - 1].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
        for k in cutlass.range_constexpr(W - 1):
            a = a + sv[k].to(Float32) * wr[k].to(Float32)
        a = a + xv.to(Float32) * wr[W - 1].to(Float32)
        if const_expr(self.silu):
            a = silu_f32(a, out.element_type.width == 32)
        _iket.pop()
        _iket.push("stores")
        for k in cutlass.range_constexpr(W - 2):
            state[bi, c, k] = sv[k + 1]
        state[bi, c, W - 2] = xv
        out[bi, c] = a.to(out.element_type)
        _iket.pop()

    @cute.kernel
    def kernel(self, x, state, wp, out):
        """Execute 2D decode kernel on grid.

        Applies boundary checks along the channel dimension.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bi, _ = cute.arch.block_idx()
        c = bidx * self.bs + tidx
        if const_expr(self.guard):
            if c < self.dim:
                self._body(x, state, wp, out, bi, c)
        else:
            self._body(x, state, wp, out, bi, c)


def _fake_rows(dtype, shape, cv):
    """Create fake tensor with row stride aligned to cv elements."""
    stride = (cute.sym_int64(divisibility=cv), 1)
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=cv * dtype.width // 8
    )


@functools.cache
def _compile_vec(dtype, wdtype, dim, width, has_bias, silu, cv, bs, tiles=1):
    """Compile and cache vectorized channel-major decode kernel."""
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    nb = cute.sym_int()
    return cute.compile(
        ConvUpdateVec(dim, width, has_bias, silu, cv, bs, tiles),
        _fake_rows(dt, (nb, dim), cv),
        _fake_rows(dt, (nb, dim * (width - 1)), math.gcd(cv * (width - 1), 16)),
        _fake_rows(wt, (1, dim * width), math.gcd(cv * width, 16)),
        _fake_rows(wt, (1, dim), cv),
        _fake_rows(dt, (nb, dim), cv),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_update_vec(
    x, conv_state, weight, bias=None, activation=None, out=None, cv=16, bs=32, tiles=1
):
    """Perform vectorized single-token causal conv1d update.

    Assigns one thread per cv channels. Accumulates in fp32 and updates conv_state in place.
    Requires dim to be divisible by cv.

    Args:
        x: Input tensor of shape (batch, dim).
        conv_state: Conv state tensor of shape (batch, dim, width - 1), updated in place.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: Optional activation name, None, "silu", or "swish".
        out: Optional output tensor of shape (batch, dim).
        cv: Number of adjacent channels per tile. Must divide dim.
        bs: Thread block size.
        tiles: Number of consecutive tiles per thread. Must divide dim // cv.

    Returns:
        Output tensor of shape (batch, dim).

    Raises:
        AssertionError: If dim is not divisible by cv * tiles.
    """
    B, D = x.shape
    W = weight.shape[1]
    if out is None:
        out = torch.empty_like(x)
    k = _compile_vec(
        x.dtype, weight.dtype, D, W, bias is not None, activation is not None, cv, bs, tiles
    )
    bb = bias if bias is not None else weight.view(-1)[:D]
    threads = B * (D // cv // tiles)
    k(x, conv_state.view(B, D * (W - 1)), weight.view(1, D * W), bb.view(1, D), out, threads)
    return out


def _fake(dtype, shape, last_static=True):
    """Create fake tensor with symbolic strides and aligned rows."""
    stride = tuple((cute.sym_int64() if i != len(shape) - 1 else 1 for i in range(len(shape))))
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=dtype.width // 8
    )


@functools.cache
def _compile_scalar(dtype, wdtype, dim, width, has_bias, silu, bs):
    """Compile and cache scalar decode kernel."""
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    nb = cute.sym_int()
    return cute.compile(
        ConvUpdateScalar(width, has_bias, silu, bs),
        _fake(dt, (nb, dim)),
        _fake(dt, (nb, dim, width - 1)),
        _fake(wt, (dim, width)),
        _fake(wt, (dim,)),
        _fake(dt, (nb, dim)),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, out=None, bs=128):
    """Perform single-token causal conv1d update.

    Updates conv_state in place and computes output elements with fp32 accumulation using one thread
    per channel.

    Args:
        x: Input tensor of shape (batch, dim).
        conv_state: Conv state tensor of shape (batch, dim, width - 1), updated in place.
        weight: Filter weights of shape (dim, width).
        bias: Optional bias tensor of shape (dim,).
        activation: Optional activation name, None, "silu", or "swish".
        out: Optional output tensor of shape (batch, dim).
        bs: Thread block size.

    Returns:
        Output tensor of shape (batch, dim).

    Raises:
        AssertionError: If activation is not None, "silu", or "swish".
    """
    assert activation in (None, "silu", "swish")
    B, D = x.shape
    W = weight.shape[1]
    if out is None:
        out = torch.empty_like(x)
    k = _compile_scalar(x.dtype, weight.dtype, D, W, bias is not None, activation is not None, bs)
    k(x, conv_state, weight, bias if bias is not None else weight.view(-1)[:D], out, B * D)
    return out


@functools.cache
def _compile_tm(dtype, wdtype, dim, width, has_bias, silu, cv, bs, vec_acc=False):
    """Compile and cache time-major decode kernel."""
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    nb = cute.sym_int()
    return cute.compile(
        ConvUpdateTimeMajor(dim, width, has_bias, silu, cv, bs, vec_acc),
        _fake_rows(dt, (nb, dim), cv),
        _fake_rows(dt, (nb, (width - 1) * dim), cv),
        _fake_rows(wt, (1, width * dim), cv),
        _fake_rows(wt, (1, dim), cv),
        _fake_rows(dt, (nb, dim), cv),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_update_tm(
    x, conv_state, weight_t, bias=None, activation=None, out=None, cv=16, bs=32, vec_acc=False
):
    """Perform single-token conv update with time-major state storage.

    Updates conv_state in place. Operates on conv_state provided as a (batch, dim, width - 1) view
    over underlying (batch, width - 1, dim) contiguous storage. Accumulates in fp32.

    Args:
        x: Input tensor of shape (batch, dim).
        conv_state: Tensor view of shape (batch, dim, width - 1) whose underlying storage of shape
            (batch, width - 1, dim) is contiguous.
        weight_t: Transposed filter weights of shape (width, dim).
        bias: Optional bias tensor of shape (dim,).
        activation: Optional activation name, None, "silu", or "swish".
        out: Optional output tensor of shape (batch, dim).
        cv: Number of channels processed per thread. Must divide dim.
        bs: Thread block size.
        vec_acc: Whether to accumulate using vector operations instead of scalar chains.

    Returns:
        Output tensor of shape (batch, dim).

    Raises:
        AssertionError: If conv_state storage is not contiguous in (batch, width - 1, dim) order.
    """
    B, D = x.shape
    W = weight_t.shape[0]
    storage = conv_state.transpose(1, 2)
    assert storage.is_contiguous(), "time-major kernel needs (B, W-1, D)-contiguous state storage"
    if out is None:
        out = torch.empty_like(x)
    k = _compile_tm(
        x.dtype, weight_t.dtype, D, W, bias is not None, activation is not None, cv, bs, vec_acc
    )
    bb = bias if bias is not None else weight_t.view(-1)[:D]
    k(x, storage.view(B, (W - 1) * D), weight_t.view(1, W * D), bb.view(1, D), out, B * (D // cv))
    return out


@functools.cache
def _compile_padded(dtype, wdtype, dim, width, has_bias, silu, bs, mbp=0):
    """Compile and cache padded decode kernel."""
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    P = 1 << (width - 1).bit_length()
    nb, nrows = (cute.sym_int(), cute.sym_int())
    return cute.compile(
        ConvUpdatePadded(dim, width, has_bias, silu, bs, mbp),
        _fake(dt, (nb, dim)),
        _fake_rows(dt, (nrows, P), P),
        _fake_rows(wt, (dim, P), P),
        _fake(wt, (dim,)),
        _fake(dt, (nb, dim)),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def pad_weight(weight):
    """Pad filter weights on the right to the next power of two.

    Args:
        weight: Filter weights of shape (dim, width).

    Returns:
        Contiguous tensor of shape (dim, P) zero-padded to the next power
        of two P >= width.
    """
    W = weight.shape[1]
    return torch.nn.functional.pad(weight, (0, (1 << (W - 1).bit_length()) - W)).contiguous()


def causal_conv1d_update_padded(
    x, conv_state, weight_p, width, bias=None, activation=None, out=None, bs=128, mbp=0
):
    """Perform single-token conv update using power-of-two padded state.

    Assigns one thread per channel. Fetches channel state and filter weight rows as aligned vector
    loads. Updates conv_state in place.

    Args:
        x: Input tensor of shape (batch, dim).
        conv_state: Contiguous conv state tensor of shape (batch, dim, P), where P is the next power
            of 2 greater than or equal to width. Updated in place.
        weight_p: Contiguous filter weights of shape (dim, P) zero-padded to P.
        width: Kernel width before padding.
        bias: Optional bias tensor of shape (dim,).
        activation: Optional activation name, None, "silu", or "swish".
        out: Optional output tensor of shape (batch, dim).
        bs: Thread block size.
        mbp: Minimum blocks per multiprocessor launch hint.

    Returns:
        Output tensor of shape (batch, dim).

    Raises:
        AssertionError: If conv_state is not contiguous or has shape other than (batch, dim, P).
    """
    B, D = x.shape
    P = weight_p.shape[1]
    assert conv_state.shape == (B, D, P) and conv_state.is_contiguous()
    if out is None:
        out = torch.empty_like(x)
    k = _compile_padded(
        x.dtype, weight_p.dtype, D, width, bias is not None, activation is not None, bs, mbp
    )
    k(
        x,
        conv_state.view(B * D, P),
        weight_p,
        bias if bias is not None else weight_p.view(-1)[:D],
        out,
        B,
    )
    return out


@functools.cache
def _compile_padded_vec(dtype, wdtype, dim, width, has_bias, silu, cv, bs):
    """Compile and cache vectorized padded decode kernel."""
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    P = 1 << (width - 1).bit_length()
    nb = cute.sym_int()
    return cute.compile(
        ConvUpdatePaddedVec(dim, width, has_bias, silu, cv, bs),
        _fake_rows(dt, (nb, dim), cv),
        _fake_rows(dt, (nb, dim * P), min(16, cv * P)),
        _fake_rows(wt, (1, dim * P), min(16, cv * P)),
        _fake_rows(wt, (1, dim), cv),
        _fake_rows(dt, (nb, dim), cv),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_update_padded_vec(
    x, conv_state, weight_p, width, bias=None, activation=None, out=None, cv=16, bs=32
):
    """Perform single-token conv update with vectorized padded state.

    Processes cv channels per thread across contiguous conv state of shape (batch, dim, P). Updates
    conv_state in place. Requires that the thread count equals the work count to prevent data races
    on in-place state updates.

    Args:
        x: Input tensor of shape (batch, dim).
        conv_state: Contiguous conv state tensor of shape (batch, dim, P), where P is the next power
            of 2 greater than or equal to width. Updated in place.
        weight_p: Contiguous filter weights of shape (dim, P) zero-padded to P.
        width: Kernel width before padding.
        bias: Optional bias tensor of shape (dim,).
        activation: Optional activation name, None, "silu", or "swish".
        out: Optional output tensor of shape (batch, dim).
        cv: Number of channels processed per thread. Must divide dim.
        bs: Thread block size.

    Returns:
        Output tensor of shape (batch, dim).

    Raises:
        AssertionError: If conv_state is not contiguous or has shape other than (batch, dim, P).
        ValueError: If (dim // cv) is not a multiple of bs.
    """
    B, D = x.shape
    P = weight_p.shape[1]
    assert conv_state.shape == (B, D, P) and conv_state.is_contiguous()
    if out is None:
        out = torch.empty_like(x)
    if D // cv % bs != 0:
        raise ValueError(
            f"the padded vector kernel needs the thread count to equal the work count: D // cv = {D // cv} is not a multiple of bs = {bs} (an out-of-range thread would race on the in-place state update)"
        )
    k = _compile_padded_vec(
        x.dtype, weight_p.dtype, D, width, bias is not None, activation is not None, cv, bs
    )
    bb = bias if bias is not None else weight_p.view(-1)[:D]
    n = B * (D // cv)
    k(
        x,
        conv_state.view(B, D * P),
        weight_p.view(1, D * P),
        bb.view(1, D),
        out,
        n,
        (n + bs - 1) // bs,
    )
    return out


@functools.cache
def _compile_std2d(dtype, wdtype, dim, width, has_bias, silu, bs):
    """Compile and cache 2D decode kernel."""
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    P = 1 << (width + int(has_bias) - 1).bit_length()
    nb = cute.sym_int()
    return cute.compile(
        ConvUpdateStd2D(dim, width, has_bias, silu, bs),
        _fake(dt, (nb, dim)),
        _fake(dt, (nb, dim, width - 1)),
        _fake_rows(wt, (dim, P), P),
        _fake(dt, (nb, dim)),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_update_std2d(
    x, conv_state, weight_packed, width, has_bias, activation=None, out=None, bs=256
):
    """Perform single-token conv update using 2D grid and packed weights.

    Uses a 2D grid over channel blocks and batch rows. Filter weights and bias are stored in the
    same row of shape (dim, P) and loaded with a single vector load. Conv state elements are
    accessed via scalar loads and stores and updated in place.

    Args:
        x: Input tensor of shape (batch, dim).
        conv_state: Contiguous conv state tensor of shape (batch, dim, width - 1), updated in place.
        weight_packed: Tensor of shape (dim, P) containing filter weights and bias stored in the
            same row, where P is the next power of 2 greater than or equal to width + int(has_bias).
        width: Kernel width.
        has_bias: Whether bias is included in weight_packed.
        activation: Optional activation name, None, "silu", or "swish".
        out: Optional output tensor of shape (batch, dim).
        bs: Thread block size.

    Returns:
        Output tensor of shape (batch, dim).

    Raises:
        AssertionError: If conv_state is not contiguous or has shape other than (batch, dim, width -
            1).
    """
    B, D = x.shape
    assert conv_state.shape == (B, D, width - 1) and conv_state.is_contiguous()
    if out is None:
        out = torch.empty_like(x)
    k = _compile_std2d(x.dtype, weight_packed.dtype, D, width, has_bias, activation is not None, bs)
    k(x, conv_state, weight_packed, out, B)
    return out
