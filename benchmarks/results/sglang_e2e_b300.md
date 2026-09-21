# End-to-End Performance: SGLang With and Without the Plugin

This document details the end-to-end performance impact of replacing SGLang's native convolution operations with the `causal-conv1d-cute` plugin. 

## Methodology & Environment

All tests were conducted on an **NVIDIA B300** GPU using the `lmsysorg/sglang:v0.5.20-cu130` Docker image (PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18, nvidia-cutlass-dsl 4.6.2). 

To ensure highly rigorous and stable measurements:
*   **Interleaved Sessions:** We ran two interleaved server sessions for both the stock setup and the plugin (e.g., Stock -> Plugin -> Stock -> Plugin) to account for hardware or system noise.
*   **Pinned Accept Length:** When testing with Multi-Token Prediction (MTP), we pinned the accept length to 3 using `SGLANG_SIMULATE_ACC_LEN=3`. Because the plugin's convolution outputs round slightly differently than the native Triton kernels, an unpinned accept length naturally drifts (e.g., from 3.01 to 3.05). Pinning it ensures we are exclusively measuring kernel execution speed rather than speculative acceptance drift.

### Server Configurations

Both the baseline and plugin arms utilized standard SGLang cookbook recipes:

**Qwen3.8-27B (bfloat16, 1 GPU):**
```bash
sglang serve --trust-remote-code --model-path Qwen/Qwen3.8-27B \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --mamba-radix-cache-strategy extra_buffer \
  [--speculative-algorithm EAGLE --speculative-num-steps 3 \
   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4]
```

**Qwen3.8-Flash-Next (FP8, 4 GPUs, Tensor Parallelism = 4):**
```bash
sglang serve --trust-remote-code --model-path Qwen/Qwen3.8-Flash-Next-FP8 --tp 4 --ep 4 \
  --mem-fraction-static 0.85 --chunked-prefill-size 8192 \
  --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer \
  --mamba-ssm-dtype bfloat16 --reasoning-parser auto \
  [--speculative-algorithm NEXTN --speculative-num-steps 3 \
   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4]
```

### Evaluated Workloads
Workloads were generated using SGLang's `bench_serving` script with a fixed seed and flushed cache. 
*   **16k Prefill:** 24 prompts (16,384 tokens in / 1 out) at concurrency 1.
*   **2k Prefill:** 64 prompts (2,048 tokens in / 1 out) at concurrency 1.
*   **Decode (c=1):** 12 requests (256 in / 512 out) at concurrency 1.
*   **Decode (c=16):** 48 requests (256 in / 1024 out) at concurrency 16.
*   **Decode (c=64):** 192 requests (256 in / 512 out) at concurrency 64.

---

## Kernel Profiling Inside the Server

Before looking at end-to-end metrics, it is vital to understand the theoretical ceiling for improvement. We used the PyTorch profiler (`/start_profile`) to measure GPU kernel activity inside the live server. 

Qwen3.8-27B utilizes 48 Gated DeltaNet layers (each with a 10,240-channel conv). Qwen3.8-Flash-Next utilizes 36 layers (2,560 channels per GPU at TP=4). When MTP is disabled, the decode phase makes a single call per layer to a fused kernel that handles the conv, Z, B, and A extractions.

| Operation Context | Stock (Triton) | Plugin | Speedup Ratio |
| --- | --- | --- | --- |
| **Qwen3.8-27B** Prefill (2048-token chunks) | 41.69 - 42.06 µs | 20.75 - 20.83 µs | **2.01x** |
| **Qwen3.8-27B** MTP Verify (c=16) | 8.47 µs | 3.97 µs | **2.13x** |
| **Qwen3.8-27B** Decode w/o MTP (c=16) | 3.43 - 4.17 µs | 2.79 - 2.83 µs | **1.35x** |
| **Qwen3.8-27B** Decode w/o MTP (c=64) | 7.71 - 7.73 µs | 4.30 µs | **1.80x** |
| **Flash-Next** Decode w/o MTP (c=64) | 3.90 - 3.91 µs | 2.82 µs | **1.38x** |
| **Flash-Next** MTP Verify (c=64) | 4.98 - 5.01 µs | 3.43 - 3.44 µs | **1.45x** |

**The Bottleneck:** In a stock Qwen3.8-27B server with MTP, the convolution accounts for roughly **2.1% of GPU time during prefill** and **1.9% during decode**. Our plugin roughly halves this footprint to 1.0% and 0.9%, respectively. Because the vast majority of server time is spent on GEMMs, delta-rule kernels, and normalization, **any end-to-end improvement is mathematically bounded to roughly 1% to 2%**. 

*(Full details available in `sglang_profile_b300.csv` and `sglang_conv_in_server_b300.csv`).*

---

## End-to-End Serving Results

As predicted by the profiler, nearly every measurement moves in the direction of the plugin by 0.1% to 2.0%. 
*(Note: Data presents averages from two interleaved sessions. Metrics like TTFT = Time To First Token; TPOT = Time Per Output Token).*

### 1. Qwen3.8-27B (1 GPU) — Multi-Token Prediction (MTP) Enabled
*Accept length strictly pinned to 3.00.*

| Workload & Metric | Stock | Plugin | Change |
| --- | --- | --- | --- |
| **Prefill 16k:** Mean Latency | 811.4 / 818.2 ms | 807.6 / 805.0 ms | **-1.0%** |
| **Prefill 16k:** Median TTFT | 805.3 / 810.6 ms | 799.6 / 797.0 ms | **-1.2%** |
| **Prefill 2k:** Mean Latency | 109.6 / 110.1 ms | 108.8 / 108.9 ms | **-0.9%** |
| **Decode (c=1):** Output tok/s | 229.8 / 229.9 | 230.7 / 230.6 | **+0.4%** |
| **Decode (c=1):** Mean TPOT | 4.28 / 4.28 ms | 4.26 / 4.26 ms | **-0.5%** |
| **Decode (c=16):** Output tok/s | 2574.5 / 2577.4 | 2591.0 / 2590.6 | **+0.6%** |
| **Decode (c=16):** Mean TPOT | 6.03 / 6.03 ms | 6.00 / 6.00 ms | **-0.5%** |

### 2. Qwen3.8-27B (1 GPU) — No MTP

| Workload & Metric | Stock | Plugin | Change |
| --- | --- | --- | --- |
| **Prefill 16k:** Median TTFT | 788.1 / 788.7 ms | 779.8 / 781.9 ms | **-1.0%** |
| **Decode (c=16):** Output tok/s | 1386.2 / 1386.3 | 1392.5 / 1392.4 | **+0.4%** |
| **Decode (c=16):** Mean TPOT | 11.38 / 11.38 ms | 11.33 / 11.33 ms | **-0.4%** |
| **Decode (c=64):** Output tok/s | 3949.9 / 3952.1 | 3989.8 / 3985.5 | **+0.9%** |
| **Decode (c=64):** Mean TPOT | 15.30 / 15.29 ms | 15.14 / 15.15 ms | **-1.0%** |

### 3. Qwen3.8-Flash-Next (FP8, TP=4) — Multi-Token Prediction (MTP) Enabled
*Accept length strictly pinned to 3.00.*

| Workload & Metric | Stock | Plugin | Change |
| --- | --- | --- | --- |
| **Prefill 16k:** Median TTFT | 287.5 / 291.9 ms | 285.5 / 284.6 ms | **-1.6%** |
| **Decode (c=16):** Output tok/s | 3654.0 / 3669.0 | 3686.1 / 3665.0 | **+0.4%** |
| **Decode (c=64):** Output tok/s | 7245.4 / 7458.1 | 7469.8 / 7525.9 | **+2.0%** |
| **Decode (c=64):** Mean TPOT | 5.45 / 5.42 ms | 5.38 / 5.37 ms | **-1.1%** |

### 4. Qwen3.8-Flash-Next (FP8, TP=4) — No MTP
*(Note: At 16k prefill for this configuration, session variance was relatively noisy. The baseline sessions naturally fluctuated by 0.7%, making the plugin's +0.6% variance statistically insignificant).*

| Workload & Metric | Stock | Plugin | Change |
| --- | --- | --- | --- |
| **Prefill 16k:** Median TTFT | 272.2 / 274.0 ms | 273.3 / 276.2 ms | *+0.6% (noise)* |
| **Decode (c=16):** Output tok/s | 2184.5 / 2191.9 | 2195.0 / 2197.2 | **+0.4%** |
| **Decode (c=64):** Output tok/s | 5645.5 / 5649.6 | 5673.4 / 5667.9 | **+0.4%** |

---

## The Systemic Side-Effects of Server Plugins

During early development, we noticed an anomaly: even though our plugin's kernel was 1.3x faster inside the server, overall decode performance (at concurrency 1) was actually **0.1% slower** than stock.

To understand why, we ran a "dry-run" control where the plugin was loaded, but operations defaulted to SGLang's native kernels. By isolating individual variables, we mapped out a fascinating systemic behavior regarding GPU memory allocation:

| Scenario (Qwen3.8-27B without MTP, c=1) | Steady Tok/s |
| --- | --- |
| Baseline (Stock) | **100.85** |
| Dry Run: Plugin allocates 12 MB of weight copies on its *first call* | 100.54 |
| Dry Run: Plugin allocates those same 12 MB *right after model load* | **100.84** |
| Dry Run: Plugin compiles/loads kernel, but allocates *nothing* | **100.85** |
| Dry Run: Stock prefill kernel, but outputs a buffer *one row larger* | 100.51 |
| **Final Plugin:** Reads model weights in-place, perfectly sized outputs | **100.97 - 101.01** |

**The Takeaway:** Identical kernels ran 0.3% slower simply because memory allocations made by the plugin shifted the location of SGLang's later buffers. When the plugin allocated memory *during* SGLang's CUDA graph capture, it broke memory alignment optimizations. Furthermore, generating an output buffer of 40 MiB *plus one row* pushed the buffer into an entirely different allocator size class. 

Our shipped plugin solves this by reading bias-free weights in-place, generating outputs that identically match the stock shape, and shifting any necessary allocations to initialization (immediately after model load). 

---

## Accuracy & Numerics

Replacing lower-level operations naturally raises questions about task performance degradation. We verified correctness across multiple boundaries:

| Check | Stock | Plugin |
| --- | --- | --- |
| **GSM8K Accuracy** (1319 questions, 5-shot) | 95.2% | 95.0% |
| **Conv States** (After prefill, verify, and decode across 88 batches) | Reference | **Bit-Identical** |
| **Fused Decode Copies** (Z, B, and A) | Reference | **Bit-Identical** |
| **Outputs within 1 ULP** of `float64` ref (using `bfloat16`) | 86.2% | **100%** |
| **Outputs within 0.5 ULP** of `float64` ref | 64.5% | **99.99%** |
| **Greedy Decoding Identical Continuations** (10 prompts) | 10 / 10 | 5 / 10 |

**Why do greedy continuations differ?** 
The plugin outputs are bit-identical to the stock states, but the numeric values differ slightly in the last bit or two. This occurs because the native Triton kernels formulate each product in `bfloat16` *before* accumulating in `float32`. Our kernels convert to `float32` *first*, making them mathematically closer to a true `float64` reference (99.99% within 0.5 ULP vs. stock's 64.5%).

While higher precision inherently alters individual greedy token generation paths (diverging in 5 out of 10 prompts), it is functionally irrelevant to macroscopic performance, as evidenced by the stable top-line GSM8K score.
