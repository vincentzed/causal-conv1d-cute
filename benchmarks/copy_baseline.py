"""Measure a device-to-device copy of each benchmarked tensor as a reference for prefill latency.

A depthwise causal conv1d must read every input and write every output once, so a plain copy of the
same tensor is the natural reference point. It is only a reference, not a proven lower bound: the
copy is the best of several copy kernels and torch.clone, and the spread across rounds is reported
so that a convolution landing within that spread of the copy is read as "indistinguishable".

The copy kernels vary the vector width, the block size and the number of vectors that one thread
moves. The last matters: a thread that moves a single vector spends most of its time on per-thread
overhead, and a copy built that way understates what the memory system can deliver.
"""

import csv
import functools
import sys

import cutlass
import cutlass.cute as cute
import harness as H
import torch
from cutlass import Int32

SHAPES = [
    ("lfm2-1.2b", 2048),
    ("lfm2-350m", 1024),
    ("nemotron3-nano-30b", 6144),
    ("nemotron3-super-tp4", 2560),
    ("glm5.3-flash-kda-tp4", 2048),
    ("kimi-k3-kda-tp8", 1536),
    ("qwen3.5-9b-gdn", 8192),
    ("qwen3.8-27b-gdn", 10240),
    ("qwen3.8-2.4t-gdn-tp8", 2560),
]
CONFIGS = [(16, bs, tiles) for tiles in (1, 2, 4, 8, 16) for bs in (64, 128, 256)]
CONFIGS += [(16, 512, 1), (8, 128, 1), (8, 256, 1)]


class Copy:
    """Copy kernel: tiles vector loads followed by tiles vector stores per thread."""

    def __init__(self, vec, bs, tiles):
        """Initialize the copy kernel.

        Args:
            vec: Number of elements per vector load and store.
            bs: Thread block size.
            tiles: Number of consecutive tiles that one thread copies.
        """
        self.vec, self.bs, self.tiles = vec, bs, tiles

    @cute.jit
    def __call__(self, x, out, nblocks: Int32, stream):
        """Launch nblocks blocks of bs threads."""
        self.kernel(x, out).launch(grid=[nblocks, 1, 1], block=[self.bs, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x, out):
        """Load tiles consecutive vectors of x into registers, then store them to out."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = (bidx * self.bs + tidx) * self.tiles
        xt = cute.logical_divide(x, (None, self.vec))
        ot = cute.logical_divide(out, (None, self.vec))
        regs = []
        for q in cutlass.range_constexpr(self.tiles):
            reg = cute.make_rmem_tensor_like(xt[0, (None, i + q)])
            cute.autovec_copy(xt[0, (None, i + q)], reg)
            regs.append(reg)
        for q in cutlass.range_constexpr(self.tiles):
            cute.autovec_copy(regs[q], ot[0, (None, i + q)])


@functools.cache
def compile_copy(vec, bs, tiles):
    """Compile the copy kernel for a vector width, block size and vectors per thread."""
    n = cute.sym_int(divisibility=vec * bs * tiles)
    fake = lambda: cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (1, n),
        stride=(cute.sym_int64(divisibility=vec), 1),
        assumed_align=vec * 2,
    )
    return cute.compile(
        Copy(vec, bs, tiles),
        fake(),
        fake(),
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def main():
    """Write one row per (model, batch) with the best copy and its spread over rounds."""
    rounds = 10
    with open(sys.argv[1], "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "model",
                "D",
                "B",
                "L",
                "bytes_moved",
                "best_copy",
                "us_copy",
                "us_copy_lo",
                "us_copy_hi",
                "us_torch_clone",
                "us_torch_clone_lo",
                "us_torch_clone_hi",
            ]
        )
        for name, D in SHAPES:
            for B, L in [(1, 2048), (8, 2048)]:
                n = B * D * L
                x = torch.randn(1, n, device="cuda", dtype=torch.bfloat16)
                out = torch.empty_like(x)
                best = None
                for vec, bs, tiles in CONFIGS:
                    if n % (vec * bs * tiles):
                        continue
                    k = compile_copy(vec, bs, tiles)
                    fn = lambda k=k, nb=n // (vec * bs * tiles): k(x, out, nb)
                    fn()
                    torch.cuda.synchronize()
                    assert torch.equal(x, out)
                    quick = H.time_us(fn, rounds=2)
                    if best is None or quick < best[0]:
                        best = (quick, vec, bs, fn, tiles)
                st = H.time_stats(best[3], rounds=rounds)
                tc = H.time_stats(lambda: out.copy_(x), rounds=rounds)
                wr.writerow(
                    [
                        name,
                        D,
                        B,
                        L,
                        4 * n,
                        f"vec{best[1]}/bs{best[2]}/tiles{best[4]}",
                        f"{st['med']:.3f}",
                        f"{st['lo']:.3f}",
                        f"{st['hi']:.3f}",
                        f"{tc['med']:.3f}",
                        f"{tc['lo']:.3f}",
                        f"{tc['hi']:.3f}",
                    ]
                )
                f.flush()
                print(
                    name,
                    B,
                    L,
                    best[1],
                    best[2],
                    best[4],
                    f"{st['med']:.2f} [{st['lo']:.2f},{st['hi']:.2f}]",
                    f"torch {tc['med']:.2f} [{tc['lo']:.2f},{tc['hi']:.2f}]",
                    flush=True,
                )


if __name__ == "__main__":
    main()
