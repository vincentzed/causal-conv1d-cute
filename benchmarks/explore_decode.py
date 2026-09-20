"""Exhaustively explore the design space of the single-token decode kernel for one layer shape.

One parametric kernel exposes every structural choice of the decode kernels in this package, and
the script times the full cross product of those choices. Every configuration is compiled, checked
against a float32 reference (the conv state must match exactly), and timed at several batch sizes
with a warm and with a cold L2 cache. One JSON line per configuration records the parameters, the
compile time, a histogram of the generated PTX instructions and every timing, so that main effects
and interactions can be analysed afterwards.

Parameters:
    cv       channels per tile, the width of the vector loads of x and out
    tiles    consecutive tiles handled by one thread
    hoist    with several tiles or rows, issue every load before the first store
    rows     batch rows handled by one thread for the same tiles; the weights are loaded once per
             thread instead of once per row (the batch size must be a multiple of it)
    bs       thread block size
    grid     "1d": flat thread index divided by the tiles per row; "2d": the batch is grid.y
    wlayout  "cm": weights (dim, 4) per channel, loaded in the widest tile that fits;
             "tm": weights (width, dim), one tile per filter position
    acc      "chain": one running sum; "tree": two partial sums that are added at the end
    layout   "std": conv state (batch, dim, width - 1), shifted every step; "pad": (batch, dim, 4),
             shifted, one aligned load; "ring": (batch, width - 1, dim), not shifted: the step
             overwrites the oldest row, so a thread stores one state tile instead of width - 1
    loads    order of the loads: state, weights, x ("swx") or x, weights, state ("xws")
    stores   state before out ("so") or out before state ("os")
    mbp      min_blocks_per_mp launch hint, 0 to leave it unset

The cross product is sharded so that several GPUs can work on it. The DSL reads its options when it
is imported and writes its PTX dumps into the directory it was started from, so every shard runs
in a directory of its own with the dumps requested from the shell:

    mkdir -p explore/ptx_0 && cd explore/ptx_0
    CUTE_DSL_KEEP=ptx CUTE_DSL_DISABLE_FILE_CACHING=1 PYTHONPATH=$REPO/benchmarks \
        python $REPO/benchmarks/explore_decode.py --out .. --shard 0 --nshards 4
"""

import argparse
import collections
import itertools
import json
import math
import time
from pathlib import Path
import cutlass
import cutlass.cute as cute
import harness as H
import torch
import torch.nn.functional as Fn
from cutlass import Float32, Int32, const_expr
from causal_conv1d_cute.fwd import silu_f32

SPACE = collections.OrderedDict(
    cv=[1, 2, 4, 8, 16],
    thread=[(1, False), (2, False), (2, True), (4, False), (4, True)],
    rows=[1],
    bs=[32, 64, 128, 256, 512],
    grid=["1d", "2d"],
    wlayout=["cm", "tm"],
    acc=["chain", "tree"],
    layout=["std", "pad"],
    loads=["swx", "xws"],
    stores=["so", "os"],
    mbp=[0, 4],
)
P = 4
POS = 1


class DecodeVariant:
    """Single-token decode kernel whose structure is fixed by a parameter dictionary."""

    def __init__(self, dim, width, p):
        """Store the layer shape and the parameters and derive the tile geometry."""
        self.dim, self.width, self.p = (dim, width, p)
        self.cv, (self.T, self.hoist), self.bs = (p["cv"], p["thread"], p["bs"])
        self.R = p.get("rows", 1)
        self.K = width - 1
        self.ring = p["layout"] == "ring"
        self.SK = P if p["layout"] == "pad" else self.K
        self.ts = self.cv if self.ring else math.gcd(self.cv * self.SK, 16)
        self.tw = math.gcd(self.cv * P, 16)
        self.gt = dim // (self.cv * self.T)
        self.guard = self.gt % self.bs != 0

    @cute.jit
    def _load(self, tile):
        """Load one tile into registers."""
        r = cute.make_rmem_tensor_like(tile)
        cute.autovec_copy(tile, r)
        return r

    @cute.jit
    def __call__(self, x, state, w, out, n: Int32, pos: Int32, nbx: Int32, nby: Int32, stream):
        """Launch nbx by nby blocks of bs threads; pos is the oldest row of a ring state."""
        if const_expr(self.p["mbp"] > 0):
            self.kernel(x, state, w, out, n, pos).launch(
                grid=[nbx, nby, 1],
                block=[self.bs, 1, 1],
                stream=stream,
                min_blocks_per_mp=self.p["mbp"],
            )
        else:
            self.kernel(x, state, w, out, n, pos).launch(
                grid=[nbx, nby, 1], block=[self.bs, 1, 1], stream=stream
            )

    def _weights(self, wt, g, pos):
        """Load the weights of tile g and return them as wv[k][c] register values.

        For a ring state the weight tensor holds one copy per position of the oldest row, with the
        filter rotated so that weight j belongs to state row j; pos selects the copy.
        """
        CV, W = (self.cv, self.width)
        if self.p["wlayout"] == "tm":
            rows = [self._load(wt[k, (None, g)]) for k in range(W)]
            return [[rows[k][c] for c in range(CV)] for k in range(W)]
        n = CV * P // self.tw
        row = pos if self.ring else 0
        tiles = [self._load(wt[row, (None, g * n + m)]) for m in range(n)]
        return [
            [tiles[(c * P + k) // self.tw][(c * P + k) % self.tw] for c in range(CV)]
            for k in range(W)
        ]

    def _state(self, st, bi, g):
        """Load the conv state of tile g and return the tiles and sv[k][c] register values."""
        CV, K, SK, TS = (self.cv, self.K, self.SK, self.ts)
        if self.ring:
            tiles = [self._load(st[bi * K + j, (None, g)]) for j in range(K)]
            return (tiles, [[tiles[j][c] for c in range(CV)] for j in range(K)])
        n = CV * SK // TS
        tiles = [self._load(st[bi, (None, g * n + m)]) for m in range(n)]
        flat = lambda c, k: c * SK + (SK - K) + k
        return (
            tiles,
            [[tiles[flat(c, k) // TS][flat(c, k) % TS] for c in range(CV)] for k in range(K)],
        )

    def _loads(self, x, state, bi, g):
        """Issue the loads of the input and the conv state of tile g of row bi."""
        xt = cute.logical_divide(x, (None, self.cv))
        st = cute.logical_divide(state, (None, self.ts))
        if self.p["loads"] == "swx":
            tiles, sv = self._state(st, bi, g)
            xr = self._load(xt[bi, (None, g)])
        else:
            xr = self._load(xt[bi, (None, g)])
            tiles, sv = self._state(st, bi, g)
        return (xr, tiles, sv)

    def _compute(self, xr, tiles, sv, wv, out, pos):
        """Compute the outputs and the new conv state of one tile in registers."""
        CV, K, SK, TS, W = (self.cv, self.K, self.SK, self.ts, self.width)
        orr = cute.make_rmem_tensor_like(xr)
        if self.ring:
            for c in range(CV):
                a = Float32(0.0)
                for j in range(K):
                    a = a + sv[j][c].to(Float32) * wv[j][c].to(Float32)
                a = a + xr[c].to(Float32) * wv[W - 1][c].to(Float32)
                orr[c] = silu_f32(a, False).to(out.element_type)
            return (orr, [xr])
        ns = [cute.make_rmem_tensor_like(tiles[0]) for _ in range(len(tiles))]
        for c in range(CV):
            terms = [(sv[k][c], wv[k][c]) for k in range(K)] + [(xr[c], wv[W - 1][c])]
            if self.p["acc"] == "chain":
                a = Float32(0.0)
                for s, wk in terms:
                    a = a + s.to(Float32) * wk.to(Float32)
            else:
                half = len(terms) // 2
                lo, hi = (Float32(0.0), Float32(0.0))
                for s, wk in terms[:half]:
                    lo = lo + s.to(Float32) * wk.to(Float32)
                for s, wk in terms[half:]:
                    hi = hi + s.to(Float32) * wk.to(Float32)
                a = lo + hi
            orr[c] = silu_f32(a, False).to(out.element_type)
            if self.p["layout"] == "std":
                new = [sv[k][c] for k in range(1, K)] + [xr[c]]
                for k in range(K):
                    ns[(c * K + k) // TS][(c * K + k) % TS] = new[k]
            else:
                old = [tiles[(c * SK + j) // TS][(c * SK + j) % TS] for j in range(1, SK)]
                for j, v in enumerate(old + [xr[c]]):
                    ns[(c * SK + j) // TS][(c * SK + j) % TS] = v
        return (orr, ns)

    def _stores(self, state, out, bi, g, orr, ns, pos):
        """Issue the stores of tile g in the configured order."""
        st = cute.logical_divide(state, (None, self.ts))
        ot = cute.logical_divide(out, (None, self.cv))
        n = len(ns)
        if self.p["stores"] == "os":
            cute.autovec_copy(orr, ot[bi, (None, g)])
        for m in range(n):
            if self.ring:
                cute.autovec_copy(ns[m], st[bi * self.K + pos, (None, g)])
            else:
                cute.autovec_copy(ns[m], st[bi, (None, g * n + m)])
        if self.p["stores"] == "so":
            cute.autovec_copy(orr, ot[bi, (None, g)])

    @cute.jit
    def _body(self, x, state, w, out, bi, gq, pos):
        """Handle the tiles and rows of one thread; bi indexes groups of rows batch rows."""
        wt = cute.logical_divide(w, (None, self.cv if self.p["wlayout"] == "tm" else self.tw))
        loaded = []
        for q in cutlass.range_constexpr(self.T):
            g = gq * self.T + q
            wv = self._weights(wt, g, pos)
            for r in cutlass.range_constexpr(self.R):
                row = bi * self.R + r
                xr, tiles, sv = self._loads(x, state, row, g)
                if const_expr(self.hoist):
                    loaded.append((row, g, xr, tiles, sv, wv))
                else:
                    orr, ns = self._compute(xr, tiles, sv, wv, out, pos)
                    self._stores(state, out, row, g, orr, ns, pos)
        if const_expr(self.hoist):
            done = [self._compute(one[2], one[3], one[4], one[5], out, pos) for one in loaded]
            for m in cutlass.range_constexpr(len(loaded)):
                self._stores(state, out, loaded[m][0], loaded[m][1], done[m][0], done[m][1], pos)

    @cute.kernel
    def kernel(self, x, state, w, out, n: Int32, pos: Int32):
        """Map a thread to a batch row and its first tile."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, by, _ = cute.arch.block_idx()
        i = bidx * self.bs + tidx
        if const_expr(self.p["grid"] == "1d"):
            bi = i // self.gt
            gq = i - bi * self.gt
            if const_expr(self.guard):
                if i < n:
                    self._body(x, state, w, out, bi, gq, pos)
            else:
                self._body(x, state, w, out, bi, gq, pos)
        else:
            if const_expr(self.guard):
                if i < self.gt:
                    self._body(x, state, w, out, by, i, pos)
            else:
                self._body(x, state, w, out, by, i, pos)


def fake(dtype, shape, align):
    """Create a fake tensor whose rows are aligned to align elements."""
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=(cute.sym_int64(divisibility=align), 1),
        assumed_align=align * dtype.width // 8,
    )


def compile_variant(dim, width, p):
    """Compile one configuration for bfloat16 tensors."""
    v = DecodeVariant(dim, width, p)
    dt = cutlass.BFloat16
    nb = cute.sym_int()
    wrows = width - 1 if v.ring else 1
    wshape, walign = ((width, dim), p["cv"]) if p["wlayout"] == "tm" else ((wrows, dim * P), v.tw)
    k = cute.compile(
        v,
        fake(dt, (nb, dim), p["cv"]),
        fake(dt, (cute.sym_int(), dim) if v.ring else (nb, dim * v.SK), v.ts),
        fake(dt, wshape, walign),
        fake(dt, (nb, dim), p["cv"]),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    return (v, k)


def runner(v, k, p, x, state, w, out):
    """Return a closure that launches one decode step on fixed tensors."""
    B, D = x.shape
    assert B % v.R == 0
    if p["grid"] == "1d":
        n = B // v.R * v.gt
        grid = ((n + v.bs - 1) // v.bs, 1)
    else:
        n = v.gt
        grid = ((v.gt + v.bs - 1) // v.bs, B // v.R)
    s2 = state.view(-1, D) if v.ring else state.view(B, -1)
    return lambda: k(x, s2, w, out, n, POS, grid[0], grid[1])


def make(dim, width, B, layout, seed=0):
    """Create the input, conv state, weights in both layouts and output for one batch size."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", dtype=torch.bfloat16, generator=g)
    x, w3 = (r(B, dim), r(dim, width) * 0.3)
    st = r(B, dim, width - 1)
    if layout == "pad":
        st = Fn.pad(st, (P - (width - 1), 0)).contiguous()
    if layout == "ring":
        st = r(B, width - 1, dim)
    wcm = Fn.pad(w3, (0, P - width)).contiguous().view(1, -1)
    if layout == "ring":
        K = width - 1
        turns = []
        for pos in range(K):
            order = [(j - pos) % K for j in range(K)] + [K]
            turns.append(Fn.pad(w3[:, order], (0, P - width)).reshape(1, -1))
        wcm = torch.cat(turns).contiguous()
    wtm = w3.t().contiguous()
    return (x, st, w3, wcm, wtm, torch.empty_like(x))


def correct(v, k, p, dim, width):
    """Check one configuration against a float32 reference on a batch of three."""
    x, st, w3, wcm, wtm, out = make(dim, width, 3 * p.get("rows", 1), p["layout"], seed=1)
    K = width - 1
    if p["layout"] == "ring":
        hist = torch.stack([st[:, (POS + k) % K] for k in range(K)], dim=-1)
        expect = st.clone()
        expect[:, POS] = x
    else:
        hist = st[..., -K:]
        expect = None
    full = torch.cat([hist.float(), x.float().unsqueeze(-1)], dim=-1)
    ref = Fn.silu((full * w3.float()).sum(-1))
    l1 = (full.abs() * w3.float().abs()).sum(-1)
    runner(v, k, p, x, st, wtm if p["wlayout"] == "tm" else wcm, out)()
    torch.cuda.synchronize()
    err = ((out.float() - ref).abs() / l1.clamp_min(1e-6)).max().item()
    same = (
        torch.equal(st, expect)
        if expect is not None
        else torch.equal(st[..., -K:], full[..., 1:].to(st.dtype))
    )
    return (err <= 1.1 * (width + 2) * 2.0**-8 and same, err)


def ptx_stats(directory, cid):
    """Summarize the PTX file of the latest compilation and keep it under the configuration id."""
    new = [f for f in directory.glob("*.ptx") if not f.stem.isdigit()]
    if not new:
        return {}
    f = max(new, key=lambda q: q.stat().st_mtime)
    ops = collections.Counter()
    for line in f.read_text().splitlines():
        t = line.strip().split()
        if t and t[0][0].isalpha() and "." in t[0]:
            ops[t[0].rstrip(";")] += 1
    keep = lambda pre: sum((c for o, c in ops.items() if o.startswith(pre)))
    f.rename(directory / f"{cid}.ptx")
    for other in new:
        if other != f:
            other.unlink()
    return {
        "lines": sum(ops.values()),
        "ld_global": keep("ld.global"),
        "st_global": keep("st.global"),
        "fma_bf16": ops.get("fma.rn.f32.bf16", 0),
        "fma_f32": ops.get("fma.rn.f32", 0),
        "cvt": keep("cvt."),
        "div": keep("div.") + keep("mul.hi"),
    }


@torch.no_grad()
def main():
    """Time this shard of the cross product and append one JSON line per configuration."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="explore")
    ap.add_argument("--dim", type=int, default=10240)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 8, 64, 256])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--space", help="JSON object that overrides the values of some parameters")
    a = ap.parse_args()
    if a.space:
        for key, values in json.loads(a.space).items():
            SPACE[key] = [tuple(v) if isinstance(v, list) else v for v in values]
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ptx = Path.cwd()
    log = root / f"shard_{a.shard}.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(line)["id"] for line in log.read_text().splitlines() if line.strip()}
    data = {(B, lay): make(a.dim, a.width, B, lay) for B in a.batches for lay in SPACE["layout"]}
    names = list(SPACE)
    todo = [c for i, c in enumerate(itertools.product(*SPACE.values())) if i % a.nshards == a.shard]
    ids = [
        i for i in range(math.prod((len(v) for v in SPACE.values()))) if i % a.nshards == a.shard
    ]
    if a.limit:
        todo, ids = (todo[: a.limit], ids[: a.limit])
    with open(log, "a") as f:
        for cid, combo in zip(ids, todo):
            if cid in done:
                continue
            p = dict(zip(names, combo))
            rec = {"id": cid, "params": p}
            try:
                t0 = time.perf_counter()
                v, k = compile_variant(a.dim, a.width, p)
                rec["compile_s"] = round(time.perf_counter() - t0, 3)
                rec["ptx"] = ptx_stats(ptx, cid)
                rec["guard"] = v.guard
                ok, err = correct(v, k, p, a.dim, a.width)
                rec["correct"], rec["bwd_err"] = (bool(ok), err)
                if ok:
                    rec["us"] = {}
                    for B in a.batches:
                        if B % p["rows"] != 0:
                            continue
                        x, st, _, wcm, wtm, out = data[B, p["layout"]]
                        fn = runner(v, k, p, x, st, wtm if p["wlayout"] == "tm" else wcm, out)
                        rec["us"][str(B)] = {
                            "warm": round(H.time_us(fn, a.rounds, False, 3, 15), 4),
                            "cold": round(H.time_us(fn, a.rounds, True, 3, 15), 4),
                        }
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200]}"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(cid, p, rec.get("us", rec.get("error", rec.get("correct"))), flush=True)


if __name__ == "__main__":
    main()
