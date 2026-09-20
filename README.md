# causal-conv1d-cute

Depthwise causal conv1d for inference, written in [CuTe DSL](https://github.com/NVIDIA/cutlass):
prefill, single-token decode, the multi-token decode step of speculative decoding, and
variable-length (packed) prefill with a conv state cache. It is the short convolution in front of
Mamba-2, Gated DeltaNet, KDA and LFM2 layers.

Tuned and measured on one NVIDIA B300. It is a pure Python package; kernels are JIT-compiled on
first use.

## Limitations

- **Inference only.** There is no backward pass. A call that needs gradients runs on a PyTorch
  fallback (native API) or is passed to the original implementation (compatibility layers).
- **One GPU architecture.** Every number here is from a B300 (sm_103). The tuned configurations in
  `tuned.json` are keyed by architecture and are not applied elsewhere; other GPUs run on
  heuristics that have not been validated.
- **bfloat16, float16 or float32**, channels a multiple of 16, kernel width 2 to 8. Anything else
  runs on the PyTorch fallback, with a warning.
- **Prefill is out of place.** Passing an output that shares storage with the input raises.
- Not implemented: `seq_idx`, `initial_states` / `return_final_states` in the batched prefill call,
  tree-structured speculative drafts. The compatibility layers pass these calls to the original op.
- **Small-batch decode is a tie, not a win.** At batch 8 and below every implementation finishes
  within a few tenths of a microsecond of the fixed cost of launching a kernel. With a cold cache
  13 of 22 such cases are ties.
- **End to end it is a small kernel.** In Qwen3.8-27B under sglang the conv is 2.1% of prefill GPU
  time and 1.3% to 2.2% of decode. Replacing it makes prefill about 1% faster and decode 0.1% (one
  request, no MTP) to 0.9% faster; on Qwen3.8-Flash-Next at TP = 4 decode is 0.4% to 2% faster
  ([details](benchmarks/results/sglang_e2e_b300.md)).

## Results

One B300, bfloat16, eleven conv shapes of real models and their tensor-parallel shards
([`benchmarks/shapes.py`](benchmarks/shapes.py)). Baselines: causal-conv1d 1.7,
flash-linear-attention 0.6, cuDNN 9.24, subquadratic-ops 0.3
([`benchmarks/BASELINES.md`](benchmarks/BASELINES.md)). Times are CUPTI kernel times under CUDA
graph replay, the median of 5 rounds of 40 replays. Cold: the L2 cache is flushed before every
replay. Warm: consecutive replays on the same buffers. Speedups are against the fastest other
implementation of each case. A cold case is a win only if our slowest round beats the best other's
fastest round; warm timings within 2% are ties.

### Kernels

| Regime | Cases | Cold W / T / L | Cold speedup | Warm W / T / L | Warm speedup |
| --- | --- | --- | --- | --- | --- |
| Prefill `[B, L, D]` | 54 | 54 / 0 / 0 | 1.07 / 1.35 / 1.50 | 54 / 0 / 0 | 1.06 / 1.45 / 1.91 |
| Prefill `[B, D, L]` | 54 | 54 / 0 / 0 | 1.06 / 1.30 / 1.75 | 53 / 1 / 0 | 1.01 / 1.50 / 2.04 |
| Decode, B ≤ 8 | 22 | 9 / 13 / 0 | 0.99 / 1.03 / 1.27 | 19 / 3 / 0 | 1.00 / 1.10 / 1.62 |
| Decode, B ≥ 64 | 22 | 22 / 0 / 0 | 1.05 / 1.33 / 2.31 | 22 / 0 / 0 | 1.07 / 1.56 / 3.16 |

Speedup columns are min / median / max ([data](benchmarks/results/kernels_b300.csv),
[summary](benchmarks/results/summary_b300.txt)).

![Prefill, channel-last](docs/figures/prefill_channel_last.png)

Figure 1 | Prefill latency relative to a device-to-device copy of the same tensor, channel-last
`[B, L, D]` storage, B = 8, L = 2048, cold cache. Right: our latency and speedup over the best other.

![Prefill, channel-first](docs/figures/prefill_contiguous.png)

Figure 2 | The same for channel-first `[B, D, L]` storage.

![Decode, warm cache](docs/figures/decode_batch_scaling_warm.png)

Figure 3 | Single-token decode latency, warm cache, conv state `[B, D, K - 1]`. K = 4 unless noted.

![Decode, cold cache](docs/figures/decode_batch_scaling.png)

Figure 4 | Single-token decode latency, cold cache.

### State layouts

Decode at large batch is bound by the bytes it moves, and the ordered `[B, D, K - 1]` state makes
every step rewrite all of it. Two optional layouts change what the caller stores, so they are
compared against every implementation including cuDNN's kernel for a padded state.

| State | A step | Cold W / T / L | Warm W / T / L | Warm speedup, B ≥ 64 |
| --- | --- | --- | --- | --- |
| Padded `[B, D, 4]` | one load, one store per tile | 28 / 16 / 0 | 41 / 3 / 0 | 1.00 / 1.34 / 2.76 |
| Ring `[B, K - 1, D]` | overwrites the oldest row only | 24 / 20 / 0 | 31 / 10 / 3 | 1.00 / 1.38 / 3.05 |

At B = 256, warm, the ring state is 1.30x to 1.38x faster than our own ordered-state kernel on the
four shapes with 4608 or more channels and no bias (Qwen3.8-27B: 4.67 µs against 6.14 µs; cuDNN
8.45 µs) and 1.15x on Nemotron-3 Nano. Where a step is not bound by bytes it ranges from 15% slower
to 12% faster, so the ordered state stays the default. Its three warm losses are 3% to 5%, one or
two timer ticks, against cuDNN's padded-state kernel at B ≤ 8.

![Decode by state layout](docs/figures/decode_state_layouts_warm.png)

Figure 5 | Decode speedup over the fastest other library (cuDNN's padded-state kernel included) for
the three state layouts, warm cache.

### SGLang

Tensor layouts recorded from a live sglang 0.5.20 server running Qwen3.8-27B, replayed against
sglang's own Triton kernels ([`benchmarks/bench_sglang.py`](benchmarks/bench_sglang.py)). Conv
kernel time, not forward-pass time. Conv states are bit-identical in every row.

| Path | Cases | Speedup |
| --- | --- | --- |
| Prefill, 16 to 8192 tokens | 12 | 1.03x to 2.17x |
| MTP verify, 1 to 48 sequences | 7 | 1.52x to 4.21x |
| Decode without MTP, 1 to 256 sequences | 9 | 1.31x to 1.82x |

Without MTP, sglang decodes with one fused kernel per layer that reads the conv input from the
`[Q | K | V | Z]` projection row, updates the conv state and copies Z, B and A into tensors of
their own; the plugin serves that call with a kernel that does the same in one launch
([`unpack.py`](causal_conv1d_cute/unpack.py)). Qwen3.8-Flash-Next at TP = 4 keeps both projections
in one buffer, which puts token rows 4120 elements apart; such rows take tiles of at most 8
channels, and on that layout prefill is 1.20x to 2.24x and decode 1.19x to 1.78x faster.

![sglang paths](docs/figures/sglang_paths.png)

Figure 6 | Conv kernel latency on the tensor layouts of Qwen3.8-27B in sglang, cold cache. Labels
give the speedup.

### End to end

Two interleaved server sessions per arm, sglang cookbook recipes, MTP accept length pinned to 3 in
both arms ([write-up](benchmarks/results/sglang_e2e_b300.md)).

| Model | Mode | Prompt latency | Time per token, 1 / 16 / 64 requests |
| --- | --- | --- | --- |
| Qwen3.8-27B, one GPU | MTP | -1.0% (16K), -0.9% (2K) | -0.4% / -0.6% / n/a |
| Qwen3.8-27B, one GPU | no MTP | -0.9% (16K) | -0.1% / -0.4% / -0.9% |
| Qwen3.8-Flash-Next FP8, TP = 4 | MTP | -1.6% (16K) | -0.9% / -0.4% / -2.0% |
| Qwen3.8-Flash-Next FP8, TP = 4 | no MTP | +0.6% (16K) | -0.4% / -0.4% / -0.4% |

The +0.6% is inside the 0.7% spread of the two stock sessions of that row. GSM8K is 95.0% with the
plugin and 95.2% without.

![Serving, Qwen3.8-27B](docs/figures/e2e_serving.png)

Figure 7 | Qwen3.8-27B in sglang: change in latency with the plugin. One point per server session,
relative to the mean of the two stock sessions.

![Serving, Qwen3.8-Flash-Next](docs/figures/e2e_serving_flash_next.png)

Figure 8 | The same for Qwen3.8-Flash-Next (FP8, TP = 4).

![Conv kernel time in the server](docs/figures/e2e_conv_kernel_in_server.png)

Figure 9 | Conv kernel latency per call inside the running server, from the torch profiler. Bars:
mean of two sessions; dots: sessions.

![Where GPU time goes](docs/figures/e2e_gpu_time_breakdown.png)

Figure 10 | Share of GPU kernel time by kernel family, Qwen3.8-27B with MTP.

## Install

```bash
pip install causal-conv1d-cute            # needs torch, nvidia-cutlass-dsl >= 4.6, apache-tvm-ffi
```

## Use

```python
import torch
from causal_conv1d_cute import causal_conv1d_fn, causal_conv1d_update, causal_conv1d_varlen_fn

w = torch.randn(2048, 4, device="cuda", dtype=torch.bfloat16)            # (dim, width)
x = torch.randn(8, 2048, 4096, device="cuda", dtype=torch.bfloat16)      # (batch, dim, seqlen)
y = causal_conv1d_fn(x, w, activation="silu")

state = torch.zeros(8, 2048, 3, device="cuda", dtype=torch.bfloat16)     # (batch, dim, width - 1)
y1 = causal_conv1d_update(x[:, :, 0].contiguous(), state, w, activation="silu")
```

`causal_conv1d_fn` also takes a `(batch, dim, seqlen)` view of `(batch, seqlen, dim)` storage, which
is what most models hold, without a copy. `causal_conv1d_update` takes `(batch, dim)` or
`(batch, dim, steps)` with optional `conv_state_indices`, and can record the conv state after every
token for speculative decoding. `causal_conv1d_varlen_fn` takes a packed `(dim, total_tokens)` input
with `query_start_loc`, `cache_indices`, `has_initial_state` and an in-place `conv_states` cache.

The ring state is opt-in, because the caller has to keep it:

```python
from causal_conv1d_cute import causal_conv1d_update_ring, to_ring, from_ring

seen = torch.full((8,), 4096, device="cuda", dtype=torch.int32)          # tokens absorbed so far
ring = to_ring(state, seen).contiguous()                                 # (batch, width - 1, dim)
y2 = causal_conv1d_update_ring(x[:, :, 1].contiguous(), ring, w, None, "silu", seen)
seen += 1
```

### Patching an existing stack

Runnable examples are in [`examples/`](examples).

**sglang.** The package registers an sglang plugin. Nothing changes until you opt in:

```bash
CAUSAL_CONV1D_CUTE_SGLANG=1 sglang serve --model-path Qwen/Qwen3.8-27B ...
```

The scheduler log then lists which calls are served and why any others are passed to sglang's own
op. Three call sites are wrapped: prefill, the decode and speculative-verify update, and the fused
decode call that a server without speculative decoding makes.
`CAUSAL_CONV1D_CUTE_SGLANG_PATHS=fn,update,unpack` selects a subset, and
`CAUSAL_CONV1D_CUTE_SGLANG_DRY_RUN=1` installs the wrappers but lets sglang's own kernels run, which
separates what the plugin's presence does to a server from what its kernels do.
[`examples/sglang_server.sh`](examples/sglang_server.sh),
[`examples/sglang_offline_engine.py`](examples/sglang_offline_engine.py).

**Code written against `causal_conv1d`** (mamba_ssm, flash-linear-attention, the fast paths of
Hugging Face transformers):

```python
from causal_conv1d_cute import dao_compat
dao_compat.install()      # after importing the model code
```

Same signatures; calls that need gradients, `seq_idx`, `initial_states` or a circular state still
reach the upstream function. [`examples/patch_upstream_api.py`](examples/patch_upstream_api.py).

**Plain `nn.Conv1d`.** [`examples/patch_torch_module.py`](examples/patch_torch_module.py) finds the
depthwise causal `nn.Conv1d` layers of a model and swaps in a module that shares their parameters.

## How it is built

**Layouts.** Weights are packed once per layer: `(dim, P)` rows with `P` a power of two, so one
aligned load fetches the filters of a channel tile, and a `(width, dim)` transpose for the
channel-last kernels. Activations are addressed as tiles of up to 16 adjacent elements, one 32-byte
load or store each, which is the widest memory transaction the instruction set has. Sums accumulate
in float32 and SiLU is `x * (0.5 * tanh(x / 2) + 0.5)` on the hardware's approximate tanh; against a
float64 reference 99.99% of bfloat16 outputs are correctly rounded.

![Numerics](docs/figures/e2e_numerics.png)

Figure 11 | Share of bfloat16 conv outputs within 0.5 and 1 ULP of a float64 reference, and GSM8K
accuracy through the running server.

**Channel-first prefill** gives each thread one tile of 16 consecutive tokens of one channel plus
the tile before it. Ragged lengths run the full tiles without a bounds check and handle the last
tile of each row in a second pass.

**Channel-last prefill** gives each thread 16 adjacent channels and walks a short strip of tokens,
issuing every load before the first store. Consecutive strips share `width - 1` tokens. When token
rows are a power of two bytes apart, which is what a serving engine produces when it passes the
leading columns of a fused projection, those shared rows collide in the cache and are evicted
before the next strip reads them; with width 4 the same convolution is 1.18x to 1.39x slower than
with any other spacing, and with width 3 up to 1.09x
([`benchmarks/row_stride.py`](benchmarks/row_stride.py)). For width 4 a thread therefore walks
several strips and keeps the shared tokens in registers, which recovers 1.16x to 1.27x; for width 3
the single strip stays faster and the tuner keeps it.

![Row stride](docs/figures/row_stride.png)

Figure 12 | Channel-last prefill of 8192 tokens against the distance between token rows of the
input, cold cache.

**Variable-length prefill** with one sequence is a single launch. The first `width - 1` outputs
depend on the old conv state, so the strip thread that covers them discards its values: it stores
them to the last row of its own span, which it then overwrites with that row's output. The thread
that owns the last strip of a channel tile also reads the old conv state, computes those outputs
and writes the new state, so each state tile is read and written by one thread. Several sequences
take a second, small launch over the sequence boundaries. The end of each sequence is read on the
device, so padded input buffers are safe.

**Inside a serving engine** a kernel is not the only thing a plugin changes. With one request in
flight, a server whose plugin passed every call to sglang's own kernels still decoded 0.3% slower
than a server without the plugin, because the plugin had allocated packed copies of the conv
weights (12 MB) while sglang was capturing its CUDA graphs, which moved every buffer allocated
after them. An output buffer one row larger than the stock op's did the same (0.34%): 40 MiB plus
one row falls into a different allocator size class than 40 MiB. Compiling and loading kernels had
no such effect. The serving kernels therefore read bias-free weights in place from the model's own
`(dim, width)` tensor, packed copies are built lazily and, where a bias makes one necessary, right
after the model is loaded, and outputs have exactly the shape the stock op allocates. Each of
these was found with a dry-run control, a server with the wrappers installed and sglang's kernels
running, which is the comparison to make before attributing a change of a few tenths of a percent
to a kernel.

**Decode** kernels were designed from a cost model of this GPU that we measured rather than
assumed. A parametric decode kernel was compiled in 13,360 configurations for one shape
([data](benchmarks/results/explore_decode_qwen3.8-27b_b300.csv)); the grid itself improved the
shipped kernel by 0% to 8%, but reading the generated programs of fast and slow configurations with
identical operation counts gave rules that did transfer:

![Design space](docs/figures/explore_decode.png)

Figure 13 | Warm decode latency of 13,360 configurations of one parametric kernel on the
Qwen3.8-27B shape. Black: best configuration with the ordered state (dashed) and with a padded
state (dotted), and the ring-state kernel (solid). Slowest 10% not shown.

1. A launch costs about 1.1 µs warm and 1.75 µs cold, plus a fraction of a nanosecond per thread
   block. With tens of thousands of blocks the block count alone sets the latency.
2. At large batch the time is bytes moved divided by bandwidth. Only moving fewer bytes helps,
   which is what the ring state does: 6 elements per channel and step instead of 8.
3. Memory transactions that one thread issues to the same cache line run one after the other;
   transactions to different lines overlap. Two adjacent 8-byte tiles per thread took 12.4 µs where
   the same tile of two batch rows took 5.8 µs, with identical instruction counts.
4. A load issued after a store waits for it. Every kernel here issues all loads first.
5. A load whose address comes from another load costs a second round trip. The first ring kernel
   looked up rotated weights by sequence position and lost 15% at batch 8 and below. It now loads
   position-independent weights alongside everything else and selects per-row weights in registers.
6. Arithmetic is nearly free next to memory transactions, except that at batch 8 and below latency
   is the latency of one thread, so threads handle 2 to 4 channels there and 4 to 8 at large batch.
7. The bias gets its own load. Packed into the weight row it doubled the row to two transactions on
   one cache line and cost 19% at batch 256.

Two properties of the toolchain shaped the code and are easy to trip over:

- In-place state updates must not map out-of-range threads onto the last element, as the
  out-of-place kernels do: a duplicate thread can read a state that its twin has already shifted.
  Those kernels launch exactly as many threads as there is work, or carry a real bounds check.
- The fused bfloat16 multiply-add is only formed in the last block of a kernel. A rarely taken
  branch placed after the hot path silently turns every multiply-add in it into a conversion plus a
  float32 multiply-add (1.7x slower, tests still pass). Conditional work goes first.

## Reproducing

```bash
pip install -e ".[dev]"   && pytest tests
pip install -e ".[bench]" && cd benchmarks
python build_dao.py                                  # upstream baseline, same nvcc flags as upstream
python bench_all.py --out results.csv --warm         # 5 rounds cold + warm, every implementation
python summarize.py results.csv
python copy_baseline.py copy.csv && python bench_sglang.py --out sglang.csv && python make_figures.py
python tune.py fwd update                            # rebuild tuned.json on your GPU
```

## Acknowledgements

Packaging, lint configuration and repository layout follow
[quack](https://github.com/Dao-AILab/quack). The reference semantics are those of
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d). The single-load padded decode state and
the tanh form of SiLU follow cuDNN frontend's causal conv1d kernels. Apache-2.0.
