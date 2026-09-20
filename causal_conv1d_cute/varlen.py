"""Execute variable-length boundary and decode convolution kernels.

Provides kernels for variable-length causal convolution:
- ConvSeqBoundary: Recomputes the first width - 1 outputs for each sequence in a variable-length
  (packed) batch after a bulk convolution pass, using initial conv state if provided, and writes
  final conv state.
- ConvUpdateIndexed: Single-token decode kernel with indexed conv state cache access for continuous
  batching.
"""

import functools
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
import torch
from .fwd import (
    _DTYPES,
    ConvFwdChannelLastStrip,
    fake_weights,
    packed_width,
    silu_f32,
    tile_weights,
)


class ConvSeqBoundary:
    """Recompute sequence boundary outputs and write final conv states.

    Performs a boundary correction pass for variable-length (packed) batches. When a bulk
    convolution pass processes packed activations across sequences, the first width - 1 outputs of
    each sequence are recomputed using the initial conv state if enabled, or zeros otherwise. The
    final conv state of each sequence is saved to the conv state cache:
    - If sequence length n >= width - 1: state contains the last width - 1 tokens.
    - If n < width - 1: state retains shifted old state and appends all n tokens. Sequences with
      cache slot equal to pad_slot_id are skipped.
    """

    def __init__(self, dim, width, has_bias, silu, has_state, bs=256):
        """Initialize boundary correction kernel parameters.

        Args:
            dim: Number of channels.
            width: Kernel width.
            has_bias: Whether filter weights include bias stored in the same row.
            silu: Whether to apply SiLU activation to outputs.
            has_state: Whether conv state caching is enabled.
            bs: Thread block size along the channel dimension. Defaults to 256.
        """
        self.dim, self.width, self.has_bias, self.silu, self.has_state, self.bs = (
            dim,
            width,
            has_bias,
            silu,
            has_state,
            bs,
        )
        self.P = packed_width(width, has_bias)

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
    def __call__(
        self, x, out, wp, qsl, cidx, hinit, state, ntok: Int32, pad: Int32, nseq: Int32, stream
    ):
        """Launch the boundary correction kernel on a CUDA stream.

        Args:
            x: Input activations of shape (channels, total_tokens).
            out: Output activations of shape (channels, total_tokens).
            wp: Filter weights and bias stored in the same row of shape (channels, packed_width).
            qsl: Cumulative sequence lengths of shape (batch + 1,).
            cidx: Conv state cache indices of shape (batch,).
            hinit: Initial conv state flags of shape (batch,).
            state: Conv state cache of shape (slots, channels, width - 1).
            ntok: Total number of tokens across sequences.
            pad: Padding slot index to skip.
            nseq: Number of sequences in the batch.
            stream: CUDA stream for launch.
        """
        self.kernel(x, out, wp, qsl, cidx, hinit, state, ntok, pad).launch(
            grid=[(self.dim + self.bs - 1) // self.bs, nseq, 1],
            block=[self.bs, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _body(self, x, out, wp, qsl, cidx, hinit, state, ntok, pad, s, c):
        """Compute boundary outputs and state for one channel and sequence.

        All loads for existing conv state, the first width - 1 tokens (head), and the last width - 1
        tokens (tail) are issued before any store. If a sequence is inactive or padded, its slot
        index is mapped to 0, and out-of-range tokens are mapped onto the last element to guarantee
        valid memory addresses.

        Computes the first width - 1 outputs per sequence using existing conv state (or zeros if
        initial state is not enabled) and sequence head tokens. Evaluates the convolution dot
        product with filter weights and optional bias in float32, applies optional float32 SiLU
        activation, and stores outputs for token indices j < n.

        When conv state tracking is enabled, records the final conv state: sequences with n >= width
        - 1 store their last width - 1 tokens; sequences with n < width - 1 retain shifted elements
        from old conv state and append sequence tokens. Sequences with n == 0 leave state unchanged.
        Updates are skipped if the sequence cache slot equals pad.

        Args:
            x: Input activations of shape (channels, total_tokens).
            out: Output activations of shape (channels, total_tokens).
            wp: Filter weights and bias stored in the same row.
            qsl: Cumulative sequence lengths of shape (batch + 1,).
            cidx: Conv state cache indices of shape (batch,).
            hinit: Initial conv state flags of shape (batch,).
            state: Conv state cache of shape (slots, channels, width - 1).
            ntok: Total number of tokens.
            pad: Padding slot index to skip.
            s: Sequence index.
            c: Channel index.
        """
        W: cutlass.Constexpr = self.width
        K: cutlass.Constexpr = self.width - 1
        P: cutlass.Constexpr = self.P
        zero16 = x[c, 0] - x[c, 0]
        bos = qsl[s]
        eos = qsl[s + 1]
        n = eos - bos
        slot = cidx[s] if const_expr(self.has_state) else Int32(0)
        live = slot != pad if const_expr(self.has_state) else n >= 0
        slot_c = slot if live else Int32(0)
        use_init = hinit[s] != 0 if const_expr(self.has_state) else n < 0
        wr = self._load(cute.logical_divide(wp, (None, P))[c, (None, 0)])
        old = []
        for k in cutlass.range_constexpr(K):
            if const_expr(self.has_state):
                v = state[slot_c, c, k]
                old.append(v if use_init else zero16)
            else:
                old.append(zero16)
        head = []
        for j in cutlass.range_constexpr(K):
            tj = bos + j
            tj = tj if tj < ntok else ntok - 1
            head.append(x[c, tj])
        tail = []
        for w in cutlass.range_constexpr(K):
            tw = eos - K + w
            tw = tw if tw >= 0 else Int32(0)
            tw = tw if tw < ntok else ntok - 1
            tail.append(x[c, tw])
        for j in cutlass.range_constexpr(K):
            a = wr[P - 1].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
            for k in cutlass.range_constexpr(W):
                rel = j - (W - 1) + k
                if const_expr(rel < 0):
                    a = a + old[K + rel].to(Float32) * wr[k].to(Float32)
                else:
                    a = a + head[rel].to(Float32) * wr[k].to(Float32)
            if const_expr(self.silu):
                a = silu_f32(a, out.element_type.width == 32)
            res = a.to(out.element_type)
            ok = live & (n > j)
            tdst = bos + j
            if ok:
                out[c, tdst] = res
        if const_expr(self.has_state):
            for w in cutlass.range_constexpr(K):
                from_x = eos - K + w >= bos
                shifted = zero16
                for q in cutlass.range_constexpr(K):
                    if const_expr(q > w):
                        shifted = old[q] if n == q - w else shifted
                val = tail[w] if from_x else shifted
                val = val if n > 0 else old[w]
                if live:
                    state[slot_c, c, w] = val

    @cute.kernel
    def kernel(self, x, out, wp, qsl, cidx, hinit, state, ntok: Int32, pad: Int32):
        """Map thread and block indices to channel and sequence.

        Args:
            x: Input activations of shape (channels, total_tokens).
            out: Output activations of shape (channels, total_tokens).
            wp: Filter weights and bias stored in the same row.
            qsl: Cumulative sequence lengths of shape (batch + 1,).
            cidx: Conv state cache indices of shape (batch,).
            hinit: Initial conv state flags of shape (batch,).
            state: Conv state cache of shape (slots, channels, width - 1).
            ntok: Total number of tokens.
            pad: Padding slot index to skip.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, s, _ = cute.arch.block_idx()
        c = bidx * self.bs + tidx
        if c < self.dim:
            self._body(x, out, wp, qsl, cidx, hinit, state, ntok, pad, s, c)


class ConvSeqBoundaryVec:
    """Boundary correction for token-major activations, sixteen channels per thread.

    Does the work of ConvSeqBoundary for a (channels, total_tokens) input that views token-major
    storage, where vec adjacent channels of one token are adjacent in memory. Each thread owns one
    channel tile of one sequence, which turns every scalar access of ConvSeqBoundary into one tile
    access. Outputs that fall beyond the end of a sequence shorter than width - 1 tokens are stored
    to the last row of the sequence before that row receives its own output, which keeps the
    stores free of branches without an extra row in the output buffer. Sequences of zero tokens
    are skipped.
    """

    def __init__(self, dim, width, has_bias, silu, xstride=None, vec=16, bs=64, wmajor="time"):
        """Initialize the kernel parameters.

        Args:
            dim: Number of channels.
            width: Kernel width.
            has_bias: Whether the time-major filter weights carry a trailing bias row.
            silu: Whether to apply SiLU activation to outputs.
            xstride: Distance in elements between consecutive token rows of the input, or None
                when the rows are dense.
            vec: Number of adjacent channels per thread.
            bs: Thread block size along the tile dimension.
            wmajor: Weight layout, "time" for packed time-major weights or "channel" for the
                caller's own bias-free (dim, width) weights read in place.
        """
        assert dim % vec == 0 and (xstride or dim) % vec == 0
        assert not (wmajor == "channel" and has_bias)
        self.wmajor = wmajor
        self.dim, self.width, self.has_bias, self.silu = (dim, width, has_bias, silu)
        self.vec, self.bs = (vec, bs)
        self.G = dim // vec
        self.XG = (xstride or dim) // vec

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
    def __call__(self, x, out, wtm, qsl, cidx, hinit, state, ntok: Int32, pad: Int32, nseq, stream):
        """Launch the kernel on a CUDA stream.

        Args:
            x: Flat view of the packed input.
            out: Flat view of the packed output.
            wtm: Time-major filter weights of shape (width + int(has_bias), channels).
            qsl: Cumulative sequence lengths of shape (batch + 1,).
            cidx: Conv state slot of each sequence, shape (batch,).
            hinit: Whether each sequence starts from its cached state, shape (batch,).
            state: Conv state cache of shape (slots, channels * (width - 1)).
            ntok: Total number of tokens.
            pad: Slot index marking sequences to skip.
            nseq: Number of sequences.
            stream: CUDA stream for launch.
        """
        self.kernel(x, out, wtm, qsl, cidx, hinit, state, ntok, pad).launch(
            grid=[(self.G + self.bs - 1) // self.bs, nseq, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.jit
    def _body(self, x, out, wtm, qsl, hinit, state, ntok, slot, s, j):
        """Recompute the first outputs of one sequence and store its final state, for one tile.

        Args:
            x: Flat view of the packed input.
            out: Flat view of the packed output.
            wtm: Time-major filter weights.
            qsl: Cumulative sequence lengths.
            hinit: Whether each sequence starts from its cached state.
            state: Conv state cache, one row per slot.
            ntok: Total number of tokens.
            slot: Conv state slot of the sequence.
            s: Sequence index.
            j: Tile index along the channel dimension.
        """
        W: cutlass.Constexpr = self.width
        K: cutlass.Constexpr = self.width - 1
        V: cutlass.Constexpr = self.vec
        G: cutlass.Constexpr = self.G
        XG: cutlass.Constexpr = self.XG
        xt = cute.logical_divide(x, (None, V))
        ot = cute.logical_divide(out, (None, V))
        st = cute.logical_divide(state, (None, V))
        zero = Float32(0.0)
        bos = qsl[s]
        eos = qsl[s + 1]
        n = eos - bos
        use = hinit[s] != 0
        wr, br = tile_weights(self, wtm, j)
        sr = [self._load(st[slot, (None, j * K + k)]) for k in range(K)]
        hr = []
        for m in cutlass.range_constexpr(K):
            row = bos + m
            row = row if row < ntok else ntok - 1
            hr.append(self._load(xt[0, (None, row * XG + j)]))
        tr = []
        for m in cutlass.range_constexpr(K):
            row = eos - K + m
            row = row if row >= 0 else Int32(0)
            row = row if row < ntok else ntok - 1
            tr.append(self._load(xt[0, (None, row * XG + j)]))
        zero16 = zero.to(x.element_type)
        old = []
        hist = []
        for k in cutlass.range_constexpr(K):
            col = [sr[(e * K + k) // V][(e * K + k) % V] if use else zero16 for e in range(V)]
            old.append(col)
            hist.append([col[e].to(Float32) for e in range(V)])
        for m in cutlass.range_constexpr(K):
            hist.append([hr[m][e].to(Float32) for e in range(V)])
        heads = []
        for t in cutlass.range_constexpr(K):
            orr = cute.make_rmem_tensor_like(hr[0])
            for e in cutlass.range_constexpr(V):
                a = br[e].to(Float32) if const_expr(self.has_bias) else zero
                for k in cutlass.range_constexpr(W):
                    a = a + hist[t + k][e] * wr[k][e].to(Float32)
                if const_expr(self.silu):
                    a = silu_f32(a, out.element_type.width == 32)
                orr[e] = a.to(out.element_type)
            heads.append(orr)
        ns = [cute.make_rmem_tensor_like(sr[0]) for _ in range(K)]
        for w in cutlass.range_constexpr(K):
            from_x = eos - K + w >= bos
            for e in cutlass.range_constexpr(V):
                shifted = zero16
                for q in cutlass.range_constexpr(K):
                    if const_expr(q > w):
                        shifted = old[q][e] if n == q - w else shifted
                val = tr[w][e] if from_x else shifted
                val = val if n > 0 else old[w][e]
                ns[(e * K + w) // V][(e * K + w) % V] = val
        for r in cutlass.range_constexpr(K):
            t: cutlass.Constexpr = K - 1 - r
            dst = bos + t if n > t else eos - 1
            cute.autovec_copy(heads[t], ot[0, (None, dst * G + j)])
        for k in cutlass.range_constexpr(K):
            cute.autovec_copy(ns[k], st[slot, (None, j * K + k)])

    @cute.kernel
    def kernel(self, x, out, wtm, qsl, cidx, hinit, state, ntok: Int32, pad: Int32):
        """Map thread and block indices to a channel tile and a sequence.

        Args:
            x: Flat view of the packed input.
            out: Flat view of the packed output.
            wtm: Time-major filter weights.
            qsl: Cumulative sequence lengths.
            cidx: Conv state slot of each sequence.
            hinit: Whether each sequence starts from its cached state.
            state: Conv state cache, one row per slot.
            ntok: Total number of tokens.
            pad: Slot index marking sequences to skip.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, s, _ = cute.arch.block_idx()
        j = bidx * self.bs + tidx
        slot = cidx[s]
        if (slot != pad) & (j < self.G) & (qsl[s + 1] > qsl[s]):
            self._body(x, out, wtm, qsl, hinit, state, ntok, slot, s, j)


class ConvFwdChannelLastStripSeq(ConvFwdChannelLastStrip):
    """Convolve one packed sequence and maintain its conv state in a single kernel launch.

    Extends the strip kernel to a sequence that starts from a cached conv state and leaves its
    last width - 1 tokens behind as the new state. The first width - 1 outputs depend on the old
    state, so the strip threads discard theirs: each stores them to the last row of its own span,
    which it then overwrites with that row's output. For each channel tile, the thread that owns the
    last strip additionally runs the boundary correction of ConvSeqBoundaryVec: it loads the old
    state, computes those outputs, and stores the new state. The state of a tile is therefore
    read and written by one thread, which orders the two accesses without a second launch. The
    end of the sequence is read from the device, so x may hold padding rows past it. The sequence
    must start at row 0, as the first entry of a cumulative length array always does.
    """

    def __init__(self, dim, width, has_bias, silu, strip, vec, bs, macro, xstride, wmajor="time"):
        """Initialize the kernel parameters.

        Args:
            dim: Number of channels.
            width: Convolution kernel width.
            has_bias: Whether the time-major filter weights carry a trailing bias row.
            silu: Whether to apply SiLU activation.
            strip: Number of tokens per strip.
            vec: Channel vector width per thread tile.
            bs: Thread block size.
            macro: Number of consecutive strips that one thread walks.
            xstride: Distance in elements between consecutive token rows of the input, or None.
            wmajor: Weight layout, "time" or "channel".
        """
        super().__init__(
            dim,
            width,
            has_bias,
            silu,
            strip,
            vec,
            bs,
            True,
            False,
            True,
            False,
            xstride,
            macro,
            wmajor,
        )
        self.seq1 = True
        self.fix = ConvSeqBoundaryVec(dim, width, has_bias, silu, xstride, vec, bs, wmajor)

    @cute.jit
    def __call__(
        self,
        x,
        wtm,
        wtail,
        out,
        qsl,
        state,
        cidx,
        hinit,
        seqlen: Int32,
        nstrips: Int32,
        nblocks: Int32,
        pad: Int32,
        stream,
    ):
        """Launch the kernel on a CUDA stream.

        Args:
            x: Flat view of the input sequence.
            wtm: Time-major filter weights of shape (width + int(has_bias), dim).
            wtail: The same weights passed a second time. The thread that handles the conv state
                loads its weights through this argument; loading them through wtm would let the
                compiler merge both loads, keep the converted weights alive into the conditional
                state block, and lose the fused bfloat16 multiply-add in every strip.
            out: Flat view of the output sequence.
            qsl: First and one-past-last token of the sequence, shape (2,).
            state: Conv state cache of shape (slots, dim * (width - 1)).
            cidx: Conv state slot of the sequence, shape (1,).
            hinit: Whether the sequence starts from its cached state, shape (1,).
            seqlen: Number of tokens.
            nstrips: Number of thread spans, of strip * macro tokens each, covering the sequence.
            nblocks: Number of thread blocks.
            pad: Slot index marking a sequence whose state must not be touched.
            stream: CUDA stream for launch.
        """
        self.kernel(x, wtm, wtail, out, qsl, state, cidx, hinit, seqlen, nstrips, pad).launch(
            grid=[nblocks, 1, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self, x, wtm, wtail, out, qsl, state, cidx, hinit, seqlen: Int32, nstrips: Int32, pad: Int32
    ):
        """Map a thread to a span of strips and a channel tile.

        Out-of-range threads repeat the strip walk of the last thread, which is harmless because the
        output does not alias the input, but only the thread whose own index owns a last strip
        touches the conv state.

        Args:
            x: Flat view of the input sequence.
            wtm: Time-major filter weights.
            wtail: The same weights, loaded by the thread that handles the conv state.
            out: Flat view of the output sequence.
            qsl: First and one-past-last token of the sequence, shape (2,).
            state: Conv state cache, one row per slot.
            cidx: Conv state slot of the sequence.
            hinit: Whether the sequence starts from its cached state.
            seqlen: Number of token rows in x, which may exceed the length of the sequence when
                the caller pads its buffers.
            nstrips: Number of thread spans, of strip * macro tokens each, covering the rows of x.
            pad: Slot index marking a sequence whose state must not be touched.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        G: cutlass.Constexpr = self.G
        raw = bidx * self.bs + tidx
        last = nstrips * G - 1
        i = raw if raw < last else last
        sidx = i // G
        j = i - sidx * G
        t0 = sidx * (self.S * self.M)
        tmax = seqlen - self.S * self.M
        t0 = t0 if t0 < tmax else tmax
        slot = cidx[0]
        if (raw <= last) & (sidx == nstrips - 1) & (slot != pad) & (qsl[1] > qsl[0]):
            self.fix._body(x, out, wtail, qsl, hinit, state, seqlen, slot, Int32(0), j)
        self._walk(x, wtm, out, Int32(0), t0, j, t0 + (self.S * self.M - 1))


class ConvUpdateIndexed:
    """Execute decode-stage causal convolution with indexed conv state cache.

    Applies single-step token updates using a conv state tensor of shape (slots, channels, width -
    1) addressed by a batch index array. Batch entries matching pad are skipped without modifying
    outputs or state rows. Filter weights and bias are stored in the same row.
    """

    def __init__(self, dim, width, has_bias, silu, bs=256):
        """Initialize indexed decode update kernel parameters.

        Args:
            dim: Number of channels.
            width: Kernel width.
            has_bias: Whether filter weights include bias stored in the same row.
            silu: Whether to apply SiLU activation to outputs.
            bs: Thread block size along the channel dimension. Defaults to 256.
        """
        self.dim, self.width, self.has_bias, self.silu, self.bs = (dim, width, has_bias, silu, bs)
        self.P = packed_width(width, has_bias)

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
    def __call__(self, x, state, wp, out, idx, pad: Int32, nb: Int32, stream):
        """Launch the indexed decode update kernel on a CUDA stream.

        Args:
            x: Input token activations of shape (batch, channels).
            state: Conv state cache of shape (slots, channels, width - 1).
            wp: Filter weights and bias stored in the same row of shape (channels, packed_width).
            out: Output activations of shape (batch, channels).
            idx: Conv state cache indices of shape (batch,).
            pad: Padding slot index to skip.
            nb: Batch size.
            stream: CUDA stream for kernel launch.
        """
        self.kernel(x, state, wp, out, idx, pad).launch(
            grid=[(self.dim + self.bs - 1) // self.bs, nb, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.jit
    def _body(self, x, state, wp, out, idx, pad, bi, c):
        """Compute decode update and state for one batch item and channel.

        If the batch entry slot equals pad, its slot index is mapped to 0 to prevent out-of-bounds
        access. Loads filter weights and historical conv state values into registers. Evaluates the
        convolution dot product with filter weights and optional bias in float32, followed by
        optional float32 SiLU activation.

        If the batch entry is not padded, shifts conv state history left by one position, stores the
        current input activation into the final state slot, and writes the result to out.

        Args:
            x: Input activations of shape (batch, channels).
            state: Conv state cache of shape (slots, channels, width - 1).
            wp: Filter weights and bias stored in the same row.
            out: Output activations of shape (batch, channels).
            idx: Conv state cache indices of shape (batch,).
            pad: Padding slot index to skip.
            bi: Batch item index.
            c: Channel index.
        """
        W: cutlass.Constexpr = self.width
        P: cutlass.Constexpr = self.P
        slot = idx[bi]
        live = slot != pad
        slot_c = slot if live else Int32(0)
        wr = self._load(cute.logical_divide(wp, (None, P))[c, (None, 0)])
        sv = [state[slot_c, c, k] for k in range(W - 1)]
        xv = x[bi, c]
        a = wr[P - 1].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
        for k in cutlass.range_constexpr(W - 1):
            a = a + sv[k].to(Float32) * wr[k].to(Float32)
        a = a + xv.to(Float32) * wr[W - 1].to(Float32)
        if const_expr(self.silu):
            a = silu_f32(a, out.element_type.width == 32)
        res = a.to(out.element_type)
        if live:
            for k in cutlass.range_constexpr(W - 2):
                state[slot_c, c, k] = sv[k + 1]
            state[slot_c, c, W - 2] = xv
            out[bi, c] = res

    @cute.kernel
    def kernel(self, x, state, wp, out, idx, pad: Int32):
        """Map thread and block indices to channel and batch item.

        Args:
            x: Input activations of shape (batch, channels).
            state: Conv state cache of shape (slots, channels, width - 1).
            wp: Filter weights and bias stored in the same row.
            out: Output activations of shape (batch, channels).
            idx: Conv state cache indices of shape (batch,).
            pad: Padding slot index to skip.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bi, _ = cute.arch.block_idx()
        c = bidx * self.bs + tidx
        if c < self.dim:
            self._body(x, state, wp, out, idx, pad, bi, c)


class ConvUpdateIndexedVec:
    """Execute indexed decode over one or more tokens, sixteen channels per thread.

    Generalizes ConvUpdateIndexed to steps tokens per sequence, as issued by speculative decoding
    when a target model verifies a chain of draft tokens. Each thread owns one tile of vec adjacent
    channels of one sequence. It loads the conv state, the steps input tokens and the filter
    weights, computes steps outputs, and then stores the outputs, the conv state after the last
    token, and optionally the conv state after every token.

    The per-token states allow the caller to roll back to any accepted prefix of the draft. Two
    layouts are supported. In the "dense" layout the states form a contiguous (slots, steps,
    channels, width - 1) tensor. In the "dedup" layout consecutive states share storage: each
    channel owns a row of steps + width - 2 elements holding the last width - 2 elements of the old
    conv state followed by the steps input tokens, and the state after token t is the window of
    width - 1 elements starting at position t.

    Batch entries whose slot equals pad are skipped. Every load is issued before any store.
    """

    def __init__(
        self,
        dim,
        width,
        has_bias,
        silu,
        steps=1,
        inter=None,
        vec=16,
        bs=64,
        tiles=1,
        wmajor="time",
    ):
        """Initialize the kernel parameters.

        Args:
            dim: Number of channels.
            width: Kernel width.
            has_bias: Whether the time-major filter weights carry a trailing bias row.
            silu: Whether to apply SiLU activation to outputs.
            steps: Number of tokens per sequence.
            inter: Layout of the per-token states: None, "dense" or "dedup".
            vec: Number of adjacent channels per tile.
            bs: Thread block size along the tile dimension.
            tiles: Number of consecutive tiles that one thread handles, one after the other.
            wmajor: Weight layout, "time" for packed time-major weights or "channel" for the
                caller's own bias-free (dim, width) weights read in place.
        """
        assert dim % (vec * tiles) == 0 and inter in (None, "dense", "dedup")
        assert not (wmajor == "channel" and has_bias)
        self.wmajor = wmajor
        self.dim, self.width, self.has_bias, self.silu = (dim, width, has_bias, silu)
        self.L, self.inter, self.vec, self.bs = (steps, inter, vec, bs)
        self.G = dim // (vec * tiles)
        self.NT = tiles
        self.K = width - 1
        self.PW = steps + width - 2
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
    def __call__(self, x, state, wtm, out, idx, inter, iidx, pad: Int32, nb: Int32, stream):
        """Launch the kernel on a CUDA stream.

        Args:
            x: Input tokens of shape (batch * steps, channels), one row per token.
            state: Conv state cache of shape (slots, channels * (width - 1)).
            wtm: Time-major filter weights of shape (width + int(has_bias), channels).
            out: Output tokens of shape (batch * steps, channels).
            idx: Conv state slot of each sequence, shape (batch,).
            inter: Per-token state cache, of shape (slots * steps, channels * (width - 1)) for the
                "dense" layout or (slots, channels * (steps + width - 2)) for the "dedup" layout.
                Ignored when the kernel was built without per-token states.
            iidx: Per-token state slot of each sequence, shape (batch,).
            pad: Slot index marking batch entries to skip.
            nb: Number of sequences.
            stream: CUDA stream for launch.
        """
        self.kernel(x, state, wtm, out, idx, inter, iidx, pad).launch(
            grid=[(self.G + self.bs - 1) // self.bs, nb, 1], block=[self.bs, 1, 1], stream=stream
        )

    @cute.jit
    def _body(self, x, state, wtm, out, inter, iidx, slot, bi, j):
        """Compute all tokens of one tile of one sequence.

        Args:
            x: Input tokens, one row per token.
            state: Conv state cache, one row per slot.
            wtm: Time-major filter weights.
            out: Output tokens, one row per token.
            inter: Per-token state cache.
            iidx: Per-token state slot of each sequence.
            slot: Conv state slot of this sequence.
            bi: Sequence index.
            j: Tile index along the channel dimension.
        """
        W: cutlass.Constexpr = self.width
        K: cutlass.Constexpr = self.K
        L: cutlass.Constexpr = self.L
        V: cutlass.Constexpr = self.vec
        PW: cutlass.Constexpr = self.PW
        xt = cute.logical_divide(x, (None, V))
        ot = cute.logical_divide(out, (None, V))
        st = cute.logical_divide(state, (None, V))
        row = bi * L
        wr, br = tile_weights(self, wtm, j)
        sr = [self._load(st[slot, (None, j * K + k)]) for k in range(K)]
        xr = [self._load(xt[row + t, (None, j)]) for t in range(L)]
        hist = [[sr[(e * K + k) // V][(e * K + k) % V] for e in range(V)] for k in range(K)]
        for t in cutlass.range_constexpr(L):
            hist.append([xr[t][e] for e in range(V)])
        outs = []
        for t in cutlass.range_constexpr(L):
            orr = cute.make_rmem_tensor_like(xr[0])
            for e in cutlass.range_constexpr(V):
                a = br[e].to(Float32) if const_expr(self.has_bias) else Float32(0.0)
                for k in cutlass.range_constexpr(W):
                    a = a + hist[t + k][e].to(Float32) * wr[k][e].to(Float32)
                if const_expr(self.silu):
                    a = silu_f32(a, out.element_type.width == 32)
                orr[e] = a.to(out.element_type)
            outs.append(orr)
        ns = [cute.make_rmem_tensor_like(sr[0]) for _ in range(K)]
        for e in cutlass.range_constexpr(V):
            for k in cutlass.range_constexpr(K):
                ns[(e * K + k) // V][(e * K + k) % V] = hist[L + k][e]
        if const_expr(self.inter == "dedup"):
            it = cute.logical_divide(inter, (None, V))
            islot = iidx[bi]
            ph = [cute.make_rmem_tensor_like(sr[0]) for _ in range(PW)]
            for e in cutlass.range_constexpr(V):
                for q in cutlass.range_constexpr(PW):
                    ph[(e * PW + q) // V][(e * PW + q) % V] = hist[1 + q][e]
            for q in cutlass.range_constexpr(PW):
                cute.autovec_copy(ph[q], it[islot, (None, j * PW + q)])
        if const_expr(self.inter == "dense"):
            it = cute.logical_divide(inter, (None, V))
            irow = iidx[bi] * L
            for t in cutlass.range_constexpr(L):
                win = [cute.make_rmem_tensor_like(sr[0]) for _ in range(K)]
                for e in cutlass.range_constexpr(V):
                    for k in cutlass.range_constexpr(K):
                        win[(e * K + k) // V][(e * K + k) % V] = hist[t + 1 + k][e]
                for k in cutlass.range_constexpr(K):
                    cute.autovec_copy(win[k], it[irow + t, (None, j * K + k)])
        for k in cutlass.range_constexpr(K):
            cute.autovec_copy(ns[k], st[slot, (None, j * K + k)])
        for t in cutlass.range_constexpr(L):
            cute.autovec_copy(outs[t], ot[row + t, (None, j)])

    @cute.kernel
    def kernel(self, x, state, wtm, out, idx, inter, iidx, pad: Int32):
        """Map thread and block indices to a channel tile and a sequence.

        Args:
            x: Input tokens, one row per token.
            state: Conv state cache, one row per slot.
            wtm: Time-major filter weights.
            out: Output tokens, one row per token.
            idx: Conv state slot of each sequence.
            inter: Per-token state cache.
            iidx: Per-token state slot of each sequence.
            pad: Slot index marking batch entries to skip.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bi, _ = cute.arch.block_idx()
        j = bidx * self.bs + tidx
        slot = idx[bi]
        if const_expr(self.guard):
            if (slot != pad) & (j < self.G):
                for q in cutlass.range_constexpr(self.NT):
                    self._body(x, state, wtm, out, inter, iidx, slot, bi, j * self.NT + q)
        else:
            if slot != pad:
                for q in cutlass.range_constexpr(self.NT):
                    self._body(x, state, wtm, out, inter, iidx, slot, bi, j * self.NT + q)


def _f(dtype, shape, align=None):
    """Create a fake tensor with symbolic strides and element alignment.

    Args:
        dtype: Element data type.
        shape: Tensor shape tuple.
        align: Assumed memory alignment in bytes. Defaults to element width.

    Returns:
        Fake tensor descriptor for compilation.
    """
    stride = tuple((cute.sym_int64() if i != len(shape) - 1 else 1 for i in range(len(shape))))
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=align or max(dtype.width // 8, 1)
    )


def _wfake(wt, dim, P):
    """Create a fake weight tensor with full-row alignment.

    Args:
        wt: Weight data type.
        dim: Channel dimension.
        P: Packed width including filter weights and optional bias.

    Returns:
        Fake tensor descriptor with row-major layout and full-row alignment.
    """
    return cute.runtime.make_fake_tensor(
        wt, (dim, P), stride=(P, 1), assumed_align=P * wt.width // 8
    )


@functools.cache
def _compile_boundary(
    dtype, wdtype, dim, width, has_bias, silu, has_state, bs=256, channel_last=False
):
    """Compile the boundary correction kernel for given configurations.

    Args:
        dtype: Activation data type string.
        wdtype: Weight data type string.
        dim: Channel dimension.
        width: Kernel width.
        has_bias: Whether filter weights include bias stored in the same row.
        silu: Whether SiLU activation is enabled.
        has_state: Whether conv states are maintained.
        bs: Thread block size along the channel dimension. Defaults to 256.
        channel_last: Whether the (channels, total_tokens) activations view token-major storage,
            with adjacent channels one element apart and a symbolic distance between tokens.

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    P = packed_width(width, has_bias)
    T, ns1, ns, nh, slots = (cute.sym_int() for _ in range(5))
    act = lambda: (
        cute.runtime.make_fake_tensor(
            dt, (dim, T), stride=(1, cute.sym_int64()), assumed_align=dt.width // 8
        )
        if channel_last
        else _f(dt, (dim, T))
    )
    return cute.compile(
        ConvSeqBoundary(dim, width, has_bias, silu, has_state, bs),
        act(),
        act(),
        _wfake(wt, dim, P),
        _f(cutlass.Int32, (ns1,)),
        _f(cutlass.Int32, (ns,)),
        _f(cutlass.Uint8, (nh,)),
        _f(dt, (slots, dim, width - 1)),
        Int32(1),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_update_indexed(dtype, wdtype, dim, width, has_bias, silu, bs=256):
    """Compile the indexed decode update kernel for given configurations.

    Args:
        dtype: Activation data type string.
        wdtype: Weight data type string.
        dim: Channel dimension.
        width: Kernel width.
        has_bias: Whether filter weights include bias stored in the same row.
        silu: Whether SiLU activation is enabled.
        bs: Thread block size along the channel dimension. Defaults to 256.

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    P = packed_width(width, has_bias)
    nb, slots = (cute.sym_int(), cute.sym_int())
    return cute.compile(
        ConvUpdateIndexed(dim, width, has_bias, silu, bs),
        _f(dt, (nb, dim)),
        _f(dt, (slots, dim, width - 1)),
        _wfake(wt, dim, P),
        _f(dt, (nb, dim)),
        _f(cutlass.Int32, (nb,)),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _rows(dtype, cols, vec):
    """Create a fake two-dimensional tensor with a symbolic row count and row stride.

    Args:
        dtype: Element data type.
        cols: Number of columns.
        vec: Channel vector width; the row stride is a multiple of it and rows are aligned to it.

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
def _compile_update_indexed_vec(
    dtype,
    wdtype,
    dim,
    width,
    has_bias,
    silu,
    steps=1,
    inter=None,
    vec=16,
    bs=64,
    tiles=1,
    wmajor="time",
):
    """Compile the tile-vectorized indexed decode kernel for given configurations.

    Args:
        dtype: Activation data type string.
        wdtype: Weight data type string.
        dim: Channel dimension.
        width: Kernel width.
        has_bias: Whether the time-major filter weights carry a trailing bias row.
        silu: Whether SiLU activation is enabled.
        steps: Number of tokens per sequence.
        inter: Layout of the per-token states: None, "dense" or "dedup".
        vec: Number of adjacent channels per tile.
        bs: Thread block size along the tile dimension.
        tiles: Number of consecutive tiles that one thread handles.
        wmajor: Weight layout, "time" or "channel".

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    K = width - 1
    icols = dim * (steps + width - 2) if inter == "dedup" else dim * K
    wfake = fake_weights(wt, dim, width, has_bias, vec, wmajor)
    return cute.compile(
        ConvUpdateIndexedVec(dim, width, has_bias, silu, steps, inter, vec, bs, tiles, wmajor),
        _rows(dt, dim, vec),
        _rows(dt, dim * K, vec),
        wfake,
        _rows(dt, dim, vec),
        _f(cutlass.Int32, (cute.sym_int(),)),
        _rows(dt, icols, vec),
        _f(cutlass.Int32, (cute.sym_int(),)),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_boundary_vec(
    dtype, wdtype, dim, width, has_bias, silu, xstride=None, vec=16, bs=64, wmajor="time"
):
    """Compile the tiled boundary correction kernel for token-major activations.

    Args:
        dtype: Activation data type string.
        wdtype: Weight data type string.
        dim: Channel dimension.
        width: Kernel width.
        has_bias: Whether the time-major filter weights carry a trailing bias row.
        silu: Whether SiLU activation is enabled.
        xstride: Distance in elements between consecutive token rows of the input, or None.
        vec: Number of adjacent channels per thread.
        bs: Thread block size along the tile dimension.
        wmajor: Weight layout, "time" or "channel".

    Returns:
        Compiled CuTe kernel executable.
    """
    dt, wt = (_DTYPES[dtype], _DTYPES[wdtype])
    flat = lambda: cute.runtime.make_fake_tensor(
        dt,
        (1, cute.sym_int(divisibility=vec)),
        stride=(cute.sym_int64(divisibility=vec), 1),
        assumed_align=vec * dt.width // 8,
    )
    wfake = fake_weights(wt, dim, width, has_bias, vec, wmajor)
    return cute.compile(
        ConvSeqBoundaryVec(dim, width, has_bias, silu, xstride, vec, bs, wmajor),
        flat(),
        flat(),
        wfake,
        _f(cutlass.Int32, (cute.sym_int(),)),
        _f(cutlass.Int32, (cute.sym_int(),)),
        _f(cutlass.Uint8, (cute.sym_int(),)),
        _rows(dt, dim * (width - 1), vec),
        Int32(1),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_strip_seq(
    dtype, wdtype, dim, width, has_bias, silu, strip, vec, bs, macro, xstride, wmajor="time"
):
    """Compile and cache the single-launch kernel for one packed sequence with a conv state.

    Args:
        dtype: Data type of input and output tensors.
        wdtype: Data type of time-major filter weights.
        dim: Number of channels.
        width: Convolution kernel width.
        has_bias: Whether weights include bias.
        silu: Whether to apply SiLU activation.
        strip: Number of tokens per strip.
        vec: Channel vector width per thread tile.
        bs: Thread block size.
        macro: Number of consecutive strips that one thread walks.
        xstride: Distance in elements between consecutive token rows of the input, or None.
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
    wfake = lambda: fake_weights(wt, dim, width, has_bias, vec, wmajor)
    state = cute.runtime.make_fake_tensor(
        dt,
        (cute.sym_int(), dim * (width - 1)),
        stride=(cute.sym_int64(divisibility=vec), 1),
        assumed_align=vec * dt.width // 8,
    )
    one = lambda t: cute.runtime.make_fake_tensor(
        t, (cute.sym_int(),), stride=(1,), assumed_align=t.width // 8
    )
    return cute.compile(
        ConvFwdChannelLastStripSeq(
            dim, width, has_bias, silu, strip, vec, bs, macro, xstride, wmajor
        ),
        flat(vec if xstride else dim),
        wfake(),
        wfake(),
        flat(dim),
        one(cutlass.Int32),
        state,
        one(cutlass.Int32),
        one(cutlass.Uint8),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def causal_conv1d_fwd_channellast_strip_seq(
    x, weight_tm, width, has_bias, activation, qsl, state, cidx, hinit, pad, strip, bs, macro
):
    """Convolve one packed sequence and update its conv state in a single kernel launch.

    Args:
        x: Input of shape (dim, seqlen) viewing token-major storage: adjacent channels are one
            element apart and tokens are x.stride(1) >= dim elements apart.
        weight_tm: Time-major packed weight tensor of shape (width + int(has_bias), dim).
        width: Convolution kernel width.
        has_bias: Whether bias values are present in weight_tm.
        activation: Activation function to apply, either "silu" or None.
        qsl: Int32 tensor of shape (2,) holding the first and one-past-last token of the sequence.
            Rows of x past the end of the sequence are padding; their outputs are undefined.
        state: Conv state cache of shape (slots, dim, width - 1), contiguous within a slot.
        cidx: Int32 tensor of shape (1,) holding the conv state slot of the sequence.
        hinit: Uint8 tensor of shape (1,) that is nonzero when the sequence starts from its state.
        pad: Slot index marking a sequence whose state must not be touched.
        strip: Number of tokens per strip.
        bs: Thread block size.
        macro: Number of consecutive strips that one thread walks.

    Returns:
        Output of shape (dim, seqlen) viewing contiguous (seqlen, dim) storage.
    """
    D, L = x.shape
    vec = 16
    buf = torch.empty((L + 1, D), dtype=x.dtype, device=x.device)
    span = strip * macro
    assert L >= span and L >= width - 1 and x.stride(0) == 1
    xstride = None if x.stride(1) == D else x.stride(1)
    n = (L - 1) * x.stride(1) + D
    nstrips = (L + span - 1) // span
    threads = nstrips * (D // vec)
    k = _compile_strip_seq(
        x.dtype,
        weight_tm.dtype,
        D,
        width,
        has_bias,
        activation is not None,
        strip,
        vec,
        bs,
        macro,
        xstride,
    )
    k(
        x.as_strided((1, n), (n, 1)),
        weight_tm,
        weight_tm,
        buf.view(1, -1),
        qsl,
        state.view(state.shape[0], -1),
        cidx,
        hinit,
        L,
        nstrips,
        (threads + bs - 1) // bs,
        int(pad),
    )
    return buf[:L].t()
