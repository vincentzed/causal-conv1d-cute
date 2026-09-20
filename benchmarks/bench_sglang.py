"""Benchmark the sglang plugin against the stock sglang ops on the layouts a live server passes.

The tensor layouts below were recorded from a running sglang v0.5.20 server for Qwen3.8-27B (TP1,
EAGLE/MTP with 4 draft tokens) by wrapping the conv entry points:

- Prefill: x is a (dim, tokens) view of the leading dim columns of the fused QKVZ projection, a
  (tokens, 16384) buffer, so tokens are 16384 elements apart. sglang routes this input to its
  Triton kernel. conv_states is contiguous (slots, dim, width - 1).
- Verify: x is a (batch, dim, 4) view of contiguous (batch, 4, dim) storage, and the per-token
  conv states live in an overlapping (slots, 4, dim, width - 1) view of a (slots, dim, 6) buffer.
- Decode without speculation is one fused Triton kernel per layer: it reads the conv input from
  the leading columns of the (batch, 16384) [Q | K | V | Z] projection, updates the conv state, and
  copies Z, B and A into tensors of their own. The "decode" rows time that kernel against the one
  the plugin serves the call with. The "decode_unfused" rows time the plain single-token update
  op, which the server uses when SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=0.
- Inkling: the width-4 short-convolution kernel that sglang ships for the Inkling models, on one
  packed (tokens, dim) sequence with no cached state and no residual, against the strip kernel.

GPU time is the median of CUDA graph replays timed with CUPTI, with the L2 cache flushed before
each replay. Host time is the wall time for one eager call to return, measured over many calls
with one synchronize at the end; it bounds the rate at which an eager prefill can be issued.

Every row also checks that both implementations leave bit-identical conv states and per-token
states, and reports the largest output difference relative to max(|y|, 1).
"""

import argparse
import csv
import time
import torch
import harness as H
import sglang.kernels.ops.mamba.causal_conv1d_triton as sgl_triton
import sglang.srt.layers.attention.mamba.causal_conv1d as sgl
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_causal_conv1d_update_contiguous as sgl_fused,
)
from sglang.kernels.ops.mamba import inkling_sconv as ink
from causal_conv1d_cute import sglang_compat as C
from causal_conv1d_cute.api import CausalConv1d

bf = torch.bfloat16
MODELS = [("qwen3.8-27b", 10240, 4, 16384, 4)]
PREFILL = [
    ("1x16", [16]),
    ("1x64", [64]),
    ("1x256", [256]),
    ("1x512", [512]),
    ("1x1024", [1024]),
    ("1x2048", [2048]),
    ("1x4096", [4096]),
    ("1x8192", [8192]),
    ("2x1024", [1024] * 2),
    ("8x256", [256] * 8),
    ("mixed", [3, 40, 900, 1, 1104]),
    ("32x64", [64] * 32),
]
BATCHES = [1, 2, 4, 8, 16, 32, 48]
DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SLOTS = 310


def host_us(fn, n=300):
    """Return the mean wall time in microseconds for one eager call to return."""
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t
    torch.cuda.synchronize()
    return dt / n * 1000000.0


def rel_diff(a, b):
    """Return the largest difference between two outputs relative to max(|a|, 1)."""
    return float(((a.float() - b.float()).abs() / a.float().abs().clamp_min(1.0)).max())


def prefill_row(name, D, W, cols, label, lens):
    """Time one prefill case and check that both implementations agree."""
    T, n = (sum(lens), len(lens))
    x = torch.randn(T, cols, device="cuda", dtype=bf)[:, :D].t()
    w = torch.randn(D, W, device="cuda", dtype=bf) * 0.3
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    cidx = (torch.randperm(SLOTS - 1, device="cuda")[:n] + 1).to(torch.int32)
    hinit = torch.tensor([i % 2 == 1 for i in range(n)], device="cuda")
    st0 = torch.randn(SLOTS, D, W - 1, device="cuda", dtype=bf)
    sa, sb = (st0.clone(), st0.clone())
    kw = dict(
        query_start_loc=qsl,
        cache_indices=cidx,
        has_initial_state=hinit,
        activation="silu",
        seq_lens_cpu=list(lens),
    )
    stock = lambda: sgl.causal_conv1d_fn(x, w, None, conv_states=sa, **kw)
    ours = lambda: C.around_fn(sgl.causal_conv1d_fn, x, w, None, conv_states=sb, **kw)
    ya, yb = (stock(), ours())
    torch.cuda.synchronize()
    a, b = (H.time_stats(stock, rounds=5), H.time_stats(ours, rounds=5))
    return [
        "prefill",
        name,
        label,
        T,
        n,
        f"{a['med']:.3f}",
        f"{b['med']:.3f}",
        f"{a['med'] / b['med']:.3f}",
        f"{host_us(stock):.1f}",
        f"{host_us(ours):.1f}",
        int(torch.equal(sa, sb)),
        f"{rel_diff(ya, yb):.4f}",
    ]


def decode_row(name, D, W, steps, B):
    """Time one decode or verify case and check that both implementations agree."""
    K = W - 1
    w = torch.randn(D, W, device="cuda", dtype=bf) * 0.3
    st0 = torch.randn(SLOTS, D, K, device="cuda", dtype=bf)
    idx = (torch.randperm(SLOTS - 1, device="cuda")[:B] + 1).to(torch.int32)
    sa, sb = (st0.clone(), st0.clone())
    kw = dict(conv_state_indices=idx)
    pa = pb = None
    if steps > 1:
        x = torch.randn(B, steps, D, device="cuda", dtype=bf).transpose(1, 2)
        PW = steps + K - 1
        pa, pb = (torch.zeros(B + 1, D, PW, device="cuda", dtype=bf) for _ in range(2))
        view = lambda p: p.as_strided((B + 1, steps, D, K), (D * PW, 1, PW, 1))
        iidx = torch.arange(B, device="cuda", dtype=torch.int32)
        ka = dict(kw, intermediate_conv_window=view(pa), intermediate_state_indices=iidx)
        kb = dict(kw, intermediate_conv_window=view(pb), intermediate_state_indices=iidx)
    else:
        x = torch.randn(B, D, device="cuda", dtype=bf)
        ka = kb = kw
    up = sgl_triton.causal_conv1d_update
    stock = lambda: up(x, sa, w, None, "silu", **ka)
    ours = lambda: C.around_update(up, x, sb, w, None, "silu", **kb)
    ya, yb = (stock(), ours())
    torch.cuda.synchronize()
    same = torch.equal(sa, sb) and (pa is None or torch.equal(pa, pb))
    a, b = (H.time_stats(stock, rounds=5), H.time_stats(ours, rounds=5))
    return [
        "verify" if steps > 1 else "decode_unfused",
        name,
        f"B={B}",
        B * steps,
        B,
        f"{a['med']:.3f}",
        f"{b['med']:.3f}",
        f"{a['med'] / b['med']:.3f}",
        "",
        "",
        int(same),
        f"{rel_diff(ya, yb):.4f}",
    ]


def inkling_row(D, act, T):
    """Time the Inkling short-convolution kernel against the strip kernel on one sequence."""
    w = torch.randn(D, 4, device="cuda", dtype=bf)
    layer = CausalConv1d(w, None, act)
    xt = torch.randn(T, D, device="cuda", dtype=bf)
    cache = torch.zeros(2, 3, D, device="cuda", dtype=bf)
    cu = torch.tensor([0, T], device="cuda", dtype=torch.int64)
    si = torch.zeros(T, device="cuda", dtype=torch.int32)
    safe = torch.zeros(1, device="cuda", dtype=torch.int64)
    mask = torch.zeros(1, 1, 1, device="cuda", dtype=torch.bool)
    box = {}

    def stock():
        """Run the Inkling kernel and keep its output for the comparison."""
        box["y"] = ink.causal_conv1d(
            xt, w, cache, mask, safe, cu, si, activation=act, use_residual=False
        )

    x3 = xt.unsqueeze(0).transpose(1, 2)
    out = torch.empty_like(xt).unsqueeze(0).transpose(1, 2)
    ours = lambda: layer.fwd(x3, out=out)
    stock()
    ours()
    torch.cuda.synchronize()
    a, b = (H.time_stats(stock, rounds=5), H.time_stats(ours, rounds=5))
    return [
        "inkling",
        f"D={D} act={act or 'none'}",
        f"1x{T}",
        T,
        1,
        f"{a['med']:.3f}",
        f"{b['med']:.3f}",
        f"{a['med'] / b['med']:.3f}",
        "",
        "",
        "",
        f"{rel_diff(box['y'], out[0].t()):.4f}",
    ]


@torch.no_grad()
def fused_decode_row(name, D, W, cols, B, head_dim=128):
    """Time the fused decode call of a server without speculation and check both agree."""
    V = cols - D
    heads = V // head_dim
    w = torch.randn(D, W, device="cuda", dtype=bf) * 0.3
    st0 = torch.randn(SLOTS, D, W - 1, device="cuda", dtype=bf)
    idx = (torch.randperm(SLOTS - 1, device="cuda")[:B] + 1).to(torch.int32)
    qkvz = torch.randn(B, cols, device="cuda", dtype=bf)
    ba = torch.randn(B, 2 * heads, device="cuda", dtype=bf)
    sa, sb = (st0.clone(), st0.clone())
    kw = dict(qkv_dim=D, v_dim=V, num_v_heads=heads, head_v_dim=head_dim, activation="silu")
    stock = lambda: sgl_fused(qkvz, ba, sa, w, None, idx, **kw)
    ours = lambda: C.around_unpack(sgl_fused, qkvz, ba, sb, w, None, idx, **kw)
    ra, rb = (stock(), ours())
    torch.cuda.synchronize()
    same = torch.equal(sa, sb) and all(torch.equal(u, v) for u, v in zip(ra[1:], rb[1:]))
    a, b = (H.time_stats(stock, rounds=5), H.time_stats(ours, rounds=5))
    return [
        "decode",
        name,
        f"B={B}",
        B,
        B,
        f"{a['med']:.3f}",
        f"{b['med']:.3f}",
        f"{a['med'] / b['med']:.3f}",
        "",
        "",
        int(same),
        f"{rel_diff(ra[0], rb[0]):.4f}",
    ]


def main():
    """Run every case and write one CSV row per case."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    C.MIN_PREFILL_TOKENS = 0
    head = [
        "path",
        "model",
        "case",
        "tokens",
        "seqs",
        "us_stock",
        "us_cute",
        "speedup",
        "host_us_stock",
        "host_us_cute",
        "states_identical",
        "out_rel_diff",
    ]
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(head)
        for name, D, W, cols, steps in MODELS:
            rows = [prefill_row(name, D, W, cols, label, lens) for label, lens in PREFILL]
            rows += [decode_row(name, D, W, steps, B) for B in BATCHES]
            rows += [fused_decode_row(name, D, W, cols, B) for B in DECODE_BATCHES]
            rows += [decode_row(name, D, W, 1, B) for B in BATCHES]
            for r in rows:
                wr.writerow(r)
                f.flush()
                print("  ".join(f"{v!s:>9}" for v in r), flush=True)
        for D in (1024, 4096, 8192):
            for act in (None, "silu"):
                for T in (512, 2048, 8192, 16384):
                    r = inkling_row(D, act, T)
                    wr.writerow(r)
                    f.flush()
                    print("  ".join(f"{v!s:>9}" for v in r), flush=True)
    print(C.stats)


if __name__ == "__main__":
    main()
