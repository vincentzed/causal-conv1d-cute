# Baselines

All benchmark results located in the `results/` directory were measured on a single NVIDIA B300 GPU (sm_103, driver 610.43.02, CUDA 13.0) running inside the `flashinfer/flashinfer-ci-cu130` Docker image. 

This document details each baseline, the exact versions tested, and the specific ways our test harness interacts with them. This ensures full transparency and reproducibility for every reported number.

| Name in CSV | Implementation | Version / Commit Measured |
| --- | --- | --- |
| `dao` | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) (CUDA) | Commit `cd81f04` (2026-08-20), built via `build_dao.py` |
| `fla_triton` | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) `fla.modules.conv.triton.ops` (Triton) | v0.6.0, commit `864a87f` (2026-09-18), Triton 3.8.0 |
| `cudnn` | `cudnn.ops.causal_conv1d` and `causal_conv1d_update` | cuDNN 9.24.0.43, frontend 1.29.0 |
| `cudnn_st4` | Same update op as `cudnn`, but called with a 4-element conv state | Same as `cudnn` |
| `cudnn_nwh` | `cudnn.ops.causal_conv1d_nwh` (channel-last prefill) | Same as `cudnn` |
| `subq_ops` | `subquadratic_ops_torch.causal_conv1d` | subquadratic-ops-torch-cu13 v0.3.0 |
| `torch_conv1d` | `torch.nn.functional.conv1d` (with `groups=dim`), followed by SiLU | PyTorch 2.14.0+cu130 |
| `cute`, `cute_pad`, `cute_ring` | This repository (ordered, padded, and ring conv states) | nvidia-cutlass-dsl 4.7.1, apache-tvm-ffi 0.1.14 |

---

## Implementation Details & Calling Conventions

To ensure fair comparisons, the test harness calls each baseline according to its specific requirements:

*   **`dao`:** We call the raw C++ extensions (`causal_conv1d_fwd` and `causal_conv1d_update`) directly, intentionally bypassing the Python autograd wrapper. The code is compiled using the upstream flags, crucially including `--use_fast_math`. This flag is highly impactful; without it, the median SiLU prefill latency is 1.30x slower. Note that this implementation only supports kernel widths of 2 to 4, and it automatically selects the channel-last path based on the input tensor's strides.
*   **`fla_triton`:** We use `causal_conv1d_fwd` for `(batch, seqlen, dim)` inputs and `causal_conv1d_update` for decoding. Unlike standard implementations that use a `width - 1` state, FLA requires a cache of the last `width` inputs formatted as `(batch, dim, width)`. Our harness automatically builds this cache from the reference state and validates the trailing `width - 1` columns afterward. Additionally, every implementation is run once for a numerical correctness check before timing begins. Because Triton performs its autotuning during this initial run, all subsequent timed replays benefit from FLA's optimized autotuned configuration.
*   **`cudnn`:** We utilize the public `cudnn.ops` entry points. cuDNN dynamically chooses between using its own native kernels and utilizing an engine-graph route; we log the chosen path for each test case in the CSV's `note` column. Note that cuDNN's native kernels are only available for a width of 4. Furthermore, the standard 3-element state forces cuDNN down a slower execution path. To account for this, the `cudnn_st4` baseline tests cuDNN using the 4-element state required to trigger its fast path. Consequently, `cudnn_st4` is only compared against `cute_pad` and `cute_ring` (which also modify the state layout) and is never compared against our drop-in `cute` kernels.
*   **`subq_ops`:** We call `causal_conv1d(x, weight, bias, activation)`. This is evaluated for prefill operations only.
*   **`torch_conv1d`:** This serves as the standard fallback reference for users without any custom extensions. Because it accumulates sums using the activation data type rather than higher precision, it is timed but explicitly flagged as `lowprec` in the results.

*Note: Any baseline that fails the initial numerical correctness check is immediately marked as `WRONG` and is excluded from the timing runs.*

---

## Timing Protocol

To guarantee accurate and stable measurements, we isolate GPU execution time from host overhead using the following protocol:

1.  **Measurement Infrastructure:** The `harness.time_stats` utility captures a single operation call within a CUDA graph. This graph is then replayed using `flashinfer.testing.bench_gpu_time_with_cupti`.
2.  **Execution Rounds:** A standard timing round consists of 10 warm-up replays followed immediately by 40 timed replays. 
3.  **Metrics:** Execution time is measured exclusively using CUPTI kernel activity timestamps directly on the GPU, completely eliminating host-side launch costs. A single "round" reports the median time of its 40 replays. The final summary tables report the median of 5 total rounds, alongside the fastest and slowest individual rounds.
4.  **Cache Scenarios:**
    *   **Cold Cache:** Before every single replay, we allocate and zero out a buffer twice the size of the GPU's L2 cache, followed by a device synchronization. This cache flush occurs *before* the timing window opens, ensuring the flush overhead is not included in the reported benchmark time. (Uses 5 rounds).
    *   **Warm Cache:** The L2 cache flush step is omitted, allowing the kernel to reuse data left in the cache from the previous replay. (Uses 3 rounds).
5.  **Memory Bandwidth Reference:** We include a `copy_baseline.py` script that times a pure device-to-device memory copy of the exact same byte count, using identical protocol rules. This serves as a useful reference point to illustrate how close a kernel is to pure data-movement speeds, though it is not a strict theoretical bound.
