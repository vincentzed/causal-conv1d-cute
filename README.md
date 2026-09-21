# causal-conv1d-cute

This package provides a highly optimized depthwise causal 1D convolution (conv1d) for inference, written in the [CuTe DSL](https://github.com/NVIDIA/cutlass). It is designed to function as the short convolution layer used at the start of architectures like Mamba-2, Gated DeltaNet, KDA, and LFM2 layers.

The library supports several key operations: prefill, single-token decode, the multi-token decode step used in speculative decoding, and variable-length (packed) prefill with a conv state cache. It is distributed as a pure Python package, meaning kernels are seamlessly JIT-compiled upon their first use. 

Currently, all kernels and configurations are tuned and measured specifically for an NVIDIA B300 GPU.

## End-to-End Inference Speedup Highlights

While the convolution is a relatively small kernel end-to-end (consuming roughly 2.1% of prefill GPU time and 1.3% to 2.2% of decode time in Qwen3.8-27B under sglang), replacing the standard implementation with this library yields measurable improvements:

*   **Qwen3.8-27B:** Prefill is approximately 1% faster. Decode is 0.1% to 0.9% faster for a single request without multi-token prediction (MTP).
*   **Qwen3.8-Flash-Next:** At Tensor Parallelism (TP) = 4, the decode phase becomes 0.4% to 2% faster. 
*   *(For detailed benchmarking metrics, see the [End-to-End Results](#end-to-end) section below or read the [detailed write-up](benchmarks/results/sglang_e2e_b300.md).)*

## Limitations

Before adopting this library, please be aware of the following constraints:

*   **Inference Only:** There is no backward pass. If a call requires gradients, the library will either fall back to a native PyTorch implementation or pass the call to the original upstream implementation via compatibility layers.
*   **Targeted GPU Architecture:** Every benchmark and tuned configuration (`tuned.json`) is based strictly on the NVIDIA B300 (sm_103). Other GPUs will rely on untested heuristics that have not been validated.
*   **Strict Tensor Requirements:** Inputs must use `bfloat16`, `float16`, or `float32` data types. The number of channels must be a multiple of 16, and the kernel width must be between 2 and 8. Any deviation triggers a warning and routes the operation to a PyTorch fallback.
*   **Out-of-Place Prefill Only:** Passing an output tensor that shares memory storage with the input tensor will raise an error.
*   **Unsupported Features:** The batched prefill call does not support `seq_idx`, `initial_states`, or `return_final_states`. Tree-structured speculative drafts are also unsupported. Compatibility layers will pass these specific calls to the original operator.
*   **Small-Batch Decode Diminishing Returns:** At batch sizes of 8 and below, the optimization is largely a tie rather than a clear win. In these cases, every implementation finishes within fractions of a microsecond of the fixed kernel launch cost. (For example, with a cold cache, 13 out of 22 small-batch cases resulted in a tie).

---

## Installation

```bash
pip install causal-conv1d-cute            # Requires torch, nvidia-cutlass-dsl >= 4.6, apache-tvm-ffi
```

---

## Usage

Runnable examples for various use cases can be found in the [`examples/`](examples) directory.

### Direct API Usage

The package provides functions for both standard and variable-length causal conv1d operations. 

```python
import torch
from causal_conv1d_cute import causal_conv1d_fn, causal_conv1d_update, causal_conv1d_varlen_fn

# Standard prefill
w = torch.randn(2048, 4, device="cuda", dtype=torch.bfloat16)            # (dim, width)
x = torch.randn(8, 2048, 4096, device="cuda", dtype=torch.bfloat16)      # (batch, dim, seqlen)
y = causal_conv1d_fn(x, w, activation="silu")

# Standard decode / update
state = torch.zeros(8, 2048, 3, device="cuda", dtype=torch.bfloat16)     # (batch, dim, width - 1)
y1 = causal_conv1d_update(x[:, :, 0].contiguous(), state, w, activation="silu")
```
*Note: `causal_conv1d_fn` can accept a `(batch, dim, seqlen)` view of `(batch, seqlen, dim)` storage—which is how most models store data—without requiring a memory copy. `causal_conv1d_update` accepts either `(batch, dim)` or `(batch, dim, steps)` alongside optional `conv_state_indices`, making it capable of recording the conv state after every token for speculative decoding.*

*For packed inputs, `causal_conv1d_varlen_fn` accepts a `(dim, total_tokens)` tensor along with `query_start_loc`, `cache_indices`, `has_initial_state`, and an in-place `conv_states` cache.*

**Using the Ring State:**
The ring state layout optimizes memory movement during decode by overwriting only the oldest row, but it is an opt-in feature because the caller must manage it.

```python
from causal_conv1d_cute import causal_conv1d_update_ring, to_ring, from_ring

seen = torch.full((8,), 4096, device="cuda", dtype=torch.int32)          # tokens absorbed so far
ring = to_ring(state, seen).contiguous()                                 # (batch, width - 1, dim)
y2 = causal_conv1d_update_ring(x[:, :, 1].contiguous(), ring, w, None, "silu", seen)
seen += 1
```

### Patching an Existing Stack

**SGLang Integration**
The package includes an SGLang plugin. It remains dormant until explicitly enabled via environment variables:

```bash
CAUSAL_CONV1D_CUTE_SGLANG=1 sglang serve --model-path Qwen/Qwen3.8-27B ...
```
Once enabled, the scheduler log will indicate which calls are handled by the plugin and which are delegated to SGLang's native operations. The plugin wraps three specific call sites: prefill, the decode/speculative-verify update, and the fused decode call used when speculative decoding is disabled. 

*   Use `CAUSAL_CONV1D_CUTE_SGLANG_PATHS=fn,update,unpack` to select a subset of these paths.
*   Use `CAUSAL_CONV1D_CUTE_SGLANG_DRY_RUN=1` to install the wrappers but still run SGLang's original kernels. This is highly recommended for isolating performance changes caused by kernel execution versus merely loading the plugin.
*   *References:* [`examples/sglang_server.sh`](examples/sglang_server.sh), [`examples/sglang_offline_engine.py`](examples/sglang_offline_engine.py).

**Patching `causal_conv1d` (Dao-AILab)**
If your code uses the upstream `causal_conv1d` (e.g., `mamba_ssm`, `flash-linear-attention`, or fast paths in Hugging Face transformers), you can hot-swap the API:

```python
from causal_conv1d_cute import dao_compat
dao_compat.install()      # Call this after importing your model code
```
Signatures remain identical. Unsupported calls (gradients, `seq_idx`, `initial_states`, or circular states) will automatically route to the upstream function. 
*   *Reference:* [`examples/patch_upstream_api.py`](examples/patch_upstream_api.py).

**Patching Plain `nn.Conv1d`**
If your model relies on standard PyTorch `nn.Conv1d` layers for its depthwise causal convolutions, you can dynamically replace them with a compatible module that shares the original parameters.
*   *Reference:* [`examples/patch_torch_module.py`](examples/patch_torch_module.py).

---

## Benchmarks & Results

All benchmarks were run on a single NVIDIA B300 using `bfloat16` precision across eleven real-world convolution shapes and their tensor-parallel shards (see [`benchmarks/shapes.py`](benchmarks/shapes.py)). 

**Baselines:** We compared against `causal-conv1d` 1.7, `flash-linear-attention` 0.6, `cuDNN` 9.24, and `subquadratic-ops` 0.3 (see [`benchmarks/BASELINES.md`](benchmarks/BASELINES.md)). 

**Methodology:** Timings reflect CUPTI kernel times measured under CUDA graph replay. The reported numbers represent the median of 5 rounds, with each round consisting of 40 replays. 
*   *Cold cache:* The L2 cache is aggressively flushed before every replay.
*   *Warm cache:* Replays are executed consecutively on the same buffers. 
*   *Win/Tie/Loss:* A "cold" case is considered a win only if our *slowest* round beats the fastest competitor's *fastest* round. For "warm" timings, any results within 2% of each other are considered a tie. Speedups denote performance relative to the fastest alternative implementation.

### Core Kernels

| Regime | Cases | Cold W / T / L | Cold speedup | Warm W / T / L | Warm speedup |
| --- | --- | --- | --- | --- | --- |
| Prefill `[B, L, D]` | 54 | 54 / 0 / 0 | 1.07 / 1.35 / 1.50 | 54 / 0 / 0 | 1.06 / 1.45 / 1.91 |
| Prefill `[B, D, L]` | 54 | 54 / 0 / 0 | 1.06 / 1.30 / 1.75 | 53 / 1 / 0 | 1.01 / 1.50 / 2.04 |
| Decode, B ≤ 8 | 22 | 9 / 13 / 0 | 0.99 / 1.03 / 1.27 | 19 / 3 / 0 | 1.00 / 1.10 / 1.62 |
| Decode, B ≥ 64 | 22 | 22 / 0 / 0 | 1.05 / 1.33 / 2.31 | 22 / 0 / 0 | 1.07 / 1.56 / 3.16 |

*(Speedup columns represent min / median / max. See [raw data](benchmarks/results/kernels_b300.csv) or [summary](benchmarks/results/summary_b300.txt).)*

![Prefill, channel-last](docs/figures/prefill_channel_last.png)  
**Figure 1 |** Prefill latency relative to a device-to-device copy of the same tensor, using channel-last `[B, L, D]` storage (B = 8, L = 2048, cold cache). Right: Our latency and speedup over the best alternative.

![Prefill, channel-first](docs/figures/prefill_contiguous.png)  
**Figure 2 |** Same comparison as Figure 1, but for channel-first `[B, D, L]` storage.

![Decode, warm cache](docs/figures/decode_batch_scaling_warm.png)  
**Figure 3 |** Single-token decode latency (warm cache). Conv state shape is `[B, D, K - 1]` where K = 4 unless otherwise noted.

![Decode, cold cache](docs/figures/decode_batch_scaling.png)  
**Figure 4 |** Single-token decode latency (cold cache).

### State Layouts for Decode

During large-batch decoding, performance is bound by memory bandwidth (the number of bytes moved). The standard ordered `[B, D, K - 1]` state layout forces the kernel to rewrite the entire state during every step. To combat this, we provide two optional alternative layouts. We compared these alternatives against all other implementations, including cuDNN's kernel which uses a padded state.

| State Layout | Work per step | Cold W / T / L | Warm W / T / L | Warm speedup, B ≥ 64 |
| --- | --- | --- | --- | --- |
| Padded `[B, D, 4]` | One load, one store per tile | 28 / 16 / 0 | 41 / 3 / 0 | 1.00 / 1.34 / 2.76 |
| Ring `[B, K - 1, D]` | Overwrites the oldest row only | 24 / 20 / 0 | 31 / 10 / 3 | 1.00 / 1.38 / 3.05 |

**Key Findings:**
*   At B = 256 (warm cache), the ring state layout is 1.30x to 1.38x faster than our own ordered-state kernel across four shapes containing 4608+ channels and no bias. For example, on Qwen3.8-27B, the ring state takes 4.67 µs compared to 6.14 µs for the ordered state (and 8.45 µs for cuDNN). On Nemotron-3 Nano, it provides a 1.15x speedup.
*   In scenarios not bottlenecked by memory bandwidth, the ring state performs anywhere from 15% slower to 12% faster. Therefore, the ordered state remains the default.
*   The ordered state's only three warm-cache losses (by a marginal 3% to 5%, equating to one or two timer ticks) occur at B ≤ 8 against cuDNN's padded-state kernel.

![Decode by state layout](docs/figures/decode_state_layouts_warm.png)  
**Figure 5 |** Decode speedup over the fastest competitor library (including cuDNN's padded-state kernel) across the three state layouts (warm cache).

### SGLang Specific Benchmarks

We recorded real tensor layouts from a live SGLang 0.5.20 server running Qwen3.8-27B and replayed them directly against SGLang's native Triton kernels (see [`benchmarks/bench_sglang.py`](benchmarks/bench_sglang.py)). Note that these times reflect the convolution kernel execution time, not the entire forward pass. The convolution states remain bit-identical across implementations.

| SGLang Path | Test Cases | Speedup |
| --- | --- | --- |
| Prefill (16 to 8192 tokens) | 12 | 1.03x to 2.17x |
| MTP verify (1 to 48 sequences) | 7 | 1.52x to 4.21x |
| Decode without MTP (1 to 256 seqs) | 9 | 1.31x to 1.82x |

**Unpacking Optimization:**
When multi-token prediction (MTP) is disabled, SGLang traditionally decodes using a single fused kernel per layer. This kernel reads the conv input from a combined `[Q | K | V | Z]` projection row, updates the conv state, and then scatters Z, B, and A into individual tensors. Our plugin seamlessly provides a replacement kernel that executes all of these operations in a single launch ([`unpack.py`](causal_conv1d_cute/unpack.py)). 

For example, on Qwen3.8-Flash-Next at TP = 4, both projections are kept in a single buffer, which places token rows 4120 elements apart. These rows process tiles of at most 8 channels. On this specific memory layout, our prefill is 1.20x to 2.24x faster, and decode is 1.19x to 1.78x faster.

![sglang paths](docs/figures/sglang_paths.png)  
**Figure 6 |** Conv kernel latency evaluated on Qwen3.8-27B tensor layouts extracted from SGLang (cold cache). Labels indicate our speedup.

### End-to-End Server Performance

To measure real-world impact, we ran two interleaved server sessions per configuration using standard SGLang cookbook recipes. The MTP accept length was pinned to 3 in all tests. ([Detailed write-up here](benchmarks/results/sglang_e2e_b300.md)).

| Model | Mode | Prompt Latency | Time per token (1 / 16 / 64 requests) |
| --- | --- | --- | --- |
| Qwen3.8-27B, 1 GPU | MTP | -1.0% (16K), -0.9% (2K) | -0.4% / -0.6% / n/a |
| Qwen3.8-27B, 1 GPU | no MTP | -0.9% (16K) | -0.1% / -0.4% / -0.9% |
| Qwen3.8-Flash-Next FP8, TP = 4 | MTP | -1.6% (16K) | -0.9% / -0.4% / -2.0% |
| Qwen3.8-Flash-Next FP8, TP = 4 | no MTP | +0.6% (16K) | -0.4% / -0.4% / -0.4% |

*(Note: The +0.6% latency increase observed is well within the 0.7% variance spread of the two baseline server sessions. Additionally, end-to-end task accuracy remains stable: GSM8K scored 95.0% with the plugin versus 95.2% without it.)*

![Serving, Qwen3.8-27B](docs/figures/e2e_serving.png)  
**Figure 7 |** End-to-end latency changes in SGLang for Qwen3.8-27B. Each point represents one server session relative to the mean of two baseline sessions.

![Serving, Qwen3.8-Flash-Next](docs/figures/e2e_serving_flash_next.png)  
**Figure 8 |** The same end-to-end latency changes for Qwen3.8-Flash-Next (FP8, TP = 4).

![Conv kernel time in the server](docs/figures/e2e_conv_kernel_in_server.png)  
**Figure 9 |** Convolution kernel latency per call inside the live server, extracted via the PyTorch profiler. (Bars: mean of two sessions; dots: individual sessions).

![Where GPU time goes](docs/figures/e2e_gpu_time_breakdown.png)  
**Figure 10 |** Breakdown of GPU kernel time share by kernel family for Qwen3.8-27B with MTP enabled.

---

## Architecture & Implementation Details

### Data Layouts and Numerics
*   **Weights:** Weights are packed once per layer into `(dim, P)` rows, where `P` is a power of two. This enables a single aligned memory load to fetch all the filters for a specific channel tile. A `(width, dim)` transpose is used for the channel-last kernels.
*   **Activations:** Activations are addressed in tiles of up to 16 adjacent elements. Each tile corresponds to a single 32-byte load or store, capitalizing on the widest memory transaction supported by the instruction set.
*   **Numerics:** Sums are accumulated in `float32`. The SiLU activation utilizes the hardware's approximate `tanh` via the formula `x * (0.5 * tanh(x / 2) + 0.5)`. When compared against a `float64` reference, 99.99% of `bfloat16` outputs are correctly rounded.

![Numerics](docs/figures/e2e_numerics.png)  
**Figure 11 |** Share of `bfloat16` outputs falling within 0.5 and 1 ULP of a `float64` reference, along with GSM8K accuracy in a live server environment.

### Prefill Strategies
*   **Channel-First Prefill:** Threads are assigned one tile consisting of 16 consecutive tokens from a single channel, plus the tile immediately preceding it. Ragged sequence lengths are handled by running full tiles without bounds checks, followed by a second pass to specifically handle the final tile of each row.
*   **Channel-Last Prefill:** Threads process 16 adjacent channels and walk down a short strip of tokens, ensuring all loads are issued before the first store. Consecutive strips share `width - 1` tokens. 
    *   *The Row Stride Issue:* When the token rows are spaced apart by a power of two bytes (common in serving engines parsing fused projections), shared rows often collide in the cache and get evicted too early. For a width of 4, this cache collision can make the convolution 1.18x to 1.39x slower. To fix this, a thread walking a width of 4 will process several strips at once, keeping shared tokens safely in registers to recover a 1.16x to 1.27x speedup. For a width of 3, a single strip remains optimal (see [`benchmarks/row_stride.py`](benchmarks/row_stride.py)).

![Row stride](docs/figures/row_stride.png)  
**Figure 12 |** Channel-last prefill performance (8192 tokens) plotted against the distance between token rows in the input buffer (cold cache).

*   **Variable-Length Prefill:** For a single sequence, this requires just one kernel launch. Because the first `width - 1` outputs rely on the old conv state, the thread managing them discards its calculated values, saves them to the last row of its span, and overwrites them with the new state. A small secondary launch manages sequence boundaries for multiple sequences. Sequence ends are read directly on the device, making padded input buffers completely safe to use.

### Integration within Serving Engines
Integrating a kernel plugin into a live server involves subtle systemic side effects. During our dry-run tests (where the plugin was loaded but SGLang's native kernels still did the work), we found the server ran 0.3% slower. This slowdown occurred because our plugin allocated 12 MB of packed conv weights during SGLang's CUDA graph capture, forcing the allocator to shift every subsequent buffer. Furthermore, allocating an output buffer just one row larger than the stock operator shifted memory into an entirely different allocator size class, causing an additional 0.34% slowdown. Kernel compilation itself had no impact. 

To resolve these issues, our serving kernels now read bias-free weights in-place directly from the model's native `(dim, width)` tensor. Packed weight copies are built lazily (or immediately upon model load, if a bias is present), and output buffers are allocated to precisely match the shape of the stock operator.

### Decode Kernel Cost Model
Our decode kernels were strictly designed around empirical measurements on the B300 GPU, rather than theoretical assumptions. We compiled a parametric decode kernel into 13,360 distinct configurations for a single shape (see [raw data](benchmarks/results/explore_decode_qwen3.8-27b_b300.csv)). Analyzing the generated assembly code across both fast and slow configurations yielded seven guiding rules:

1.  **Launch Overhead Sets the Floor:** A kernel launch costs ~1.1 µs warm and ~1.75 µs cold, plus a fraction of a nanosecond per thread block. If you have tens of thousands of blocks, block count dictates latency entirely.
2.  **Memory Bandwidth Rules Large Batches:** At large batch sizes, latency equals bytes moved divided by bandwidth. The only way to go faster is to move fewer bytes. This is precisely why the ring state layout succeeds (moving 6 elements per channel/step instead of 8).
3.  **Transaction Overlap:** Memory transactions issued by a single thread to the *same* cache line run sequentially; transactions to *different* lines overlap.
4.  **Load/Store Dependencies:** A load issued after a store will stall and wait for it. Consequently, our kernels issue all loads upfront.
5.  **Pointer Chasing is Costly:** A load whose address depends on the result of another load requires a second memory round trip. Our initial ring kernel looked up rotated weights based on sequence position and lost 15% performance at batch ≤ 8. We solved this by loading position-independent weights globally and selecting per-row weights locally via registers.
6.  **Arithmetic is Cheap:** Arithmetic operations are virtually free compared to memory transactions. However, at batch ≤ 8, overall latency is bound by single-thread latency. Therefore, threads handle 2 to 4 channels for small batches, but scale up to 4 to 8 channels for large batches.
7.  **Isolate the Bias:** Packing the bias into the weight row doubled the transactions on a single cache line, costing 19% performance at batch 256. Bias loads are now handled separately.

![Design space](docs/figures/explore_decode.png)  
**Figure 13 |** Warm decode latency across 13,360 configurations of a parametric kernel (Qwen3.8-27B shape). Black markers denote the best configurations for the ordered state (dashed), padded state (dotted), and the new ring-state kernel (solid). The slowest 10% are excluded.

### Toolchain Quirks
Two compiler behaviors heavily influenced the code structure:
*   **Bounds Checking on In-Place Updates:** Unlike out-of-place kernels, in-place state updates cannot simply map out-of-range threads to the last array element. Doing so allows duplicate threads to read a state that a valid "twin" thread has already shifted. Our kernels therefore launch exactly as many threads as there is work, or employ strict, explicit bounds checks.
*   **Fused Multiply-Add Generation:** The compiler only forms the fused `bfloat16` multiply-add instruction in the final block of a kernel. Placing a rarely taken branch after the hot path silently forces the compiler to convert all multiply-adds into a slow conversion followed by a `float32` multiply-add (which is 1.7x slower, though tests still pass). To prevent this, all conditional work is evaluated early.

---

## Reproducing the Results

To run the tests and reproduce the benchmark figures locally:

```bash
# Install development and benchmark dependencies
pip install -e ".[dev]"   && pytest tests
pip install -e ".[bench]" && cd benchmarks

# Run baselines and benchmarks
python build_dao.py                                  # Builds upstream baseline (uses identical nvcc flags)
python bench_all.py --out results.csv --warm         # Runs 5 cold + warm rounds across all implementations
python summarize.py results.csv

# Generate SGLang benchmarks and figures
python copy_baseline.py copy.csv 
python bench_sglang.py --out sglang.csv 
python make_figures.py

# Rebuild tuned.json for your specific GPU architecture
python tune.py fwd update                            
```

---

## Acknowledgements

*   Packaging, linting configuration, and repository structure are based on [quack](https://github.com/Dao-AILab/quack).
*   The reference operator semantics are sourced from [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d). 
*   The single-load padded decode state and the `tanh` approximation for SiLU are inspired by the cuDNN frontend's causal conv1d kernels. 

**License:** Apache-2.0.
