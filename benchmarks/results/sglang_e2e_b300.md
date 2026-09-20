# End-to-end: sglang with and without the plugin

sglang 0.5.20 (`lmsysorg/sglang:v0.5.20-cu130`: torch 2.13.0+cu130, Triton 3.7.1, flashinfer 0.6.18,
nvidia-cutlass-dsl 4.6.2) on NVIDIA B300. Both arms of every comparison run the sglang cookbook
recipe of the model on the same GPUs; the only difference is `CAUSAL_CONV1D_CUTE_SGLANG=1`. Every
comparison is two server sessions per arm, interleaved stock, plugin, stock, plugin. With MTP,
`SGLANG_SIMULATE_ACC_LEN=3` pins the accept length in both arms, because the plugin's conv outputs
round differently from the Triton kernels' and an unpinned accept length moves with that (3.01 to
3.05 in an earlier run), which has nothing to do with kernel speed.

```bash
# Qwen3.8-27B, bfloat16, one GPU
sglang serve --trust-remote-code --model-path Qwen/Qwen3.8-27B \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --mamba-radix-cache-strategy extra_buffer \
  [--speculative-algorithm EAGLE --speculative-num-steps 3 \
   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4]

# Qwen3.8-Flash-Next, FP8 checkpoint, four GPUs
sglang serve --trust-remote-code --model-path Qwen/Qwen3.8-Flash-Next-FP8 --tp 4 --ep 4 \
  --mem-fraction-static 0.85 --chunked-prefill-size 8192 \
  --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer \
  --mamba-ssm-dtype bfloat16 --reasoning-parser auto \
  [--speculative-algorithm NEXTN --speculative-num-steps 3 \
   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4]
```

Workloads: `python -m sglang.bench_serving --backend sglang --dataset-name random
--random-range-ratio 1.0 --seed 1 --warmup-requests 3 --flush-cache` with 24 prompts of 16384 tokens
and 1 output token at concurrency 1; 64 prompts of 2048 tokens likewise; 12 requests of 256 in / 512
out at concurrency 1; 48 requests of 256 in / 1024 out at concurrency 16; 192 requests of 256 in /
512 out at concurrency 64.

## What the conv costs

Kernel time from the torch profiler inside the running server (`/start_profile`, GPU activities
only). Qwen3.8-27B has 48 Gated DeltaNet layers with one conv over 10240 channels each;
Qwen3.8-Flash-Next has 36 with 2560 channels per GPU at TP = 4. Without MTP, decode makes one call
per layer to a fused kernel that also copies Z, B and A out of the projection, in both arms.

| conv call in the running server | stock (Triton) | plugin | ratio |
| --- | --- | --- | --- |
| Qwen3.8-27B, prefill, 2048-token chunks | 41.69 - 42.06 us | 20.75 - 20.83 us | 2.01x |
| Qwen3.8-27B, MTP verify, 16 concurrent | 8.47 us | 3.97 us | 2.13x |
| Qwen3.8-27B, decode without MTP, 16 concurrent | 3.43 - 4.17 us | 2.79 - 2.83 us | 1.35x |
| Qwen3.8-27B, decode without MTP, 64 concurrent | 7.71 - 7.73 us | 4.30 us | 1.80x |
| Qwen3.8-Flash-Next, decode without MTP, 64 concurrent | 3.90 - 3.91 us | 2.82 us | 1.38x |
| Qwen3.8-Flash-Next, MTP verify, 64 concurrent | 4.98 - 5.01 us | 3.43 - 3.44 us | 1.45x |

In the stock Qwen3.8-27B server with MTP the conv is 2.09 - 2.10% of GPU kernel time in
prefill and 1.87% in decode; with the plugin 1.04% and
0.89 - 0.90%. That share bounds what any replacement can return end to end. The rest of
the step is GEMMs, the delta-rule kernels and normalization; the full breakdown is in
`sglang_profile_b300.csv` and the per-call conv times are in `sglang_conv_in_server_b300.csv`.

## Serving

### Qwen3.8-27B with MTP, accept length pinned to 3

| workload and metric | stock, two sessions | plugin, two sessions | change |
| --- | --- | --- | --- |
| 16384-token prompts, mean latency | 811.4 / 818.2 ms | 807.6 / 805.0 ms | -1.0% |
| 16384-token prompts, median TTFT | 805.3 / 810.6 ms | 799.6 / 797.0 ms | -1.2% |
| 2048-token prompts, mean latency | 109.6 / 110.1 ms | 108.8 / 108.9 ms | -0.9% |
| 1 request, output throughput | 229.8 / 229.9 tok/s | 230.7 / 230.6 tok/s | +0.4% |
| 1 request, mean TPOT | 4.28 / 4.28 ms | 4.26 / 4.26 ms | -0.5% |
| 16 concurrent, output throughput | 2574.5 / 2577.4 tok/s | 2591.0 / 2590.6 tok/s | +0.6% |
| 16 concurrent, mean TPOT | 6.03 / 6.03 ms | 6.00 / 6.00 ms | -0.5% |
| accept length, 16 concurrent | 3.00 / 3.00  | 3.00 / 3.00  | +0.0% |

### Qwen3.8-27B without MTP

| workload and metric | stock, two sessions | plugin, two sessions | change |
| --- | --- | --- | --- |
| 16384-token prompts, mean latency | 793.5 / 795.4 ms | 786.9 / 788.3 ms | -0.9% |
| 16384-token prompts, median TTFT | 788.1 / 788.7 ms | 779.8 / 781.9 ms | -1.0% |
| 1 request, output throughput | 100.34 / 100.32 tok/s | 100.41 / 100.38 tok/s | +0.1% |
| 1 request, mean TPOT | 9.92 / 9.92 ms | 9.91 / 9.91 ms | -0.1% |
| 16 concurrent, output throughput | 1386.2 / 1386.3 tok/s | 1392.5 / 1392.4 tok/s | +0.4% |
| 16 concurrent, mean TPOT | 11.38 / 11.38 ms | 11.33 / 11.33 ms | -0.4% |
| 64 concurrent, output throughput | 3949.9 / 3952.1 tok/s | 3989.8 / 3985.5 tok/s | +0.9% |
| 64 concurrent, mean TPOT | 15.30 / 15.29 ms | 15.14 / 15.15 ms | -1.0% |

### Qwen3.8-Flash-Next (FP8, TP = 4) without MTP

| workload and metric | stock, two sessions | plugin, two sessions | change |
| --- | --- | --- | --- |
| 16384-token prompts, median TTFT | 272.2 / 274.0 ms | 273.3 / 276.2 ms | +0.6% |
| 1 request, output throughput | 245.7 / 245.7 tok/s | 246.6 / 246.6 tok/s | +0.4% |
| 1 request, mean TPOT | 3.92 / 3.92 ms | 3.91 / 3.91 ms | -0.3% |
| 16 concurrent, output throughput | 2184.5 / 2191.9 tok/s | 2195.0 / 2197.2 tok/s | +0.4% |
| 16 concurrent, mean TPOT | 6.62 / 6.62 ms | 6.60 / 6.60 ms | -0.3% |
| 64 concurrent, output throughput | 5645.5 / 5649.6 tok/s | 5673.4 / 5667.9 tok/s | +0.4% |
| 64 concurrent, mean TPOT | 10.69 / 10.69 ms | 10.65 / 10.65 ms | -0.4% |

### Qwen3.8-Flash-Next (FP8, TP = 4) with MTP, accept length pinned to 3

| workload and metric | stock, two sessions | plugin, two sessions | change |
| --- | --- | --- | --- |
| 16384-token prompts, median TTFT | 287.5 / 291.9 ms | 285.5 / 284.6 ms | -1.6% |
| 1 request, output throughput | 491.2 / 490.7 tok/s | 495.8 / 495.4 tok/s | +0.9% |
| 1 request, mean TPOT | 1.87 / 1.87 ms | 1.85 / 1.85 ms | -1.1% |
| 16 concurrent, output throughput | 3654.0 / 3669.0 tok/s | 3686.1 / 3665.0 tok/s | +0.4% |
| 16 concurrent, mean TPOT | 3.64 / 3.64 ms | 3.64 / 3.64 ms | +0.0% |
| 64 concurrent, output throughput | 7245.4 / 7458.1 tok/s | 7469.8 / 7525.9 tok/s | +2.0% |
| 64 concurrent, mean TPOT | 5.45 / 5.42 ms | 5.38 / 5.37 ms | -1.1% |
| accept length, 64 concurrent | 3.00 / 3.00  | 3.00 / 3.00  | +0.0% |

Every decode row moves in the direction of the plugin, by 0.1% to 2%, which is what exchanging a
kernel that is 1% to 2% of GPU time can return. The two stock sessions of a comparison agree to 0.1%
on most decode rows. The 16384-token rows are noisier: the two stock sessions of Flash-Next without
MTP are 0.7% apart, and the plugin's +0.6% there is inside that spread, as is its -0.6% in an
earlier pair of sessions; on Qwen3.8-27B, where the conv is 2% of prefill, the 1% gain is resolved.

## What a plugin does to a server besides exchanging kernels

With one request in flight, the first version of the plugin left the decode step 0.1% slower than
stock although its kernel was 1.3x faster per call inside that same server. A dry-run control
(`CAUSAL_CONV1D_CUTE_SGLANG_DRY_RUN=1`: wrappers installed, sglang's own kernels running) showed why.
Steady single-request decode rate of Qwen3.8-27B without MTP, from the server log, reproducible to
0.02 tok/s across sessions:

| server | tok/s |
| --- | --- |
| stock | 100.85 |
| dry run; plugin allocates 12 MB of packed weight copies on its first call | 100.54 |
| dry run; the same copies allocated right after model load | 100.84 |
| dry run; the plugin only compiles and loads its kernel, allocates nothing | 100.85 |
| dry run; stock prefill kernel, plus an output buffer of one extra row as the plugin's prefill used | 100.51 |
| plugin, decode kernel reading the model's own weights, no copies | 100.97 - 101.01 |
| plugin, all three call sites, no copies, outputs of the stock size | 100.96 |

Identical kernels ran 0.3% slower because allocations made by the plugin moved the server's later
buffers: packed weight copies allocated while sglang captured its CUDA graphs, and a prefill output
of 40 MiB plus one row, which falls into a different allocator size class than the 40 MiB the stock
op allocates. The serving kernels now read bias-free weights in place, build packed copies lazily
(and right after model load where a bias makes one necessary), and allocate outputs of exactly the
stock shape.

## Accuracy

| check | stock | plugin |
| --- | --- | --- |
| GSM8K, 1319 questions, 5-shot, `sglang.test.few_shot_gsm8k`, Qwen3.8-27B with MTP | 95.2% | 95.0% |
| conv states after prefill, MTP verify and fused decode, 88 randomized batches | reference | bit-identical |
| Z, B and A copies of the fused decode call | reference | bit-identical |
| conv outputs within 0.5 ulp of a float64 reference (bfloat16) | 64.5% | 99.99% |
| conv outputs within 1 ulp of a float64 reference | 86.2% | 100% |
| greedy decoding, 10 prompts, identical token sequences vs. stock | 10 / 10 (stock vs. stock) | 5 / 10 |

The conv states that the plugin writes are bit-identical to the ones sglang's Triton kernels write
(`tests/test_sglang_plugin.py`: padded and strided inputs, mixed sequence lengths, pad slots, the
overlapping per-token window cache, projections that share one buffer). The outputs differ in the
last bit or two because the Triton kernels form each product in bfloat16 before accumulating in
float32, while these kernels convert to float32 first. That is enough to change individual greedy
continuations, as any change of kernel is, and not enough to move GSM8K. GSM8K was measured with an
earlier build of the plugin whose kernels compute the same values.
