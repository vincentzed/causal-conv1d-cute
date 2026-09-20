# Baselines

Every number in `results/` was measured on one NVIDIA B300 (sm_103, driver 610.43.02, CUDA 13.0)
inside the `flashinfer/flashinfer-ci-cu130` image. This file records what each baseline is, which
version was measured, and exactly how the harness calls it, so that a number can be challenged.

| Name in the CSV | Implementation | Version measured |
| --- | --- | --- |
| `dao` | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d), CUDA | commit `cd81f04` (2026-08-20), built by `build_dao.py` |
| `fla_triton` | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) `fla.modules.conv.triton.ops`, Triton | 0.6.0, commit `864a87f` (2026-09-18), Triton 3.8.0 |
| `cudnn` | `cudnn.ops.causal_conv1d` and `causal_conv1d_update` | cuDNN 9.24.0.43, frontend 1.29.0 |
| `cudnn_st4` | the same update op, called with a 4-element conv state | same |
| `cudnn_nwh` | `cudnn.ops.causal_conv1d_nwh` (channel-last prefill) | same |
| `subq_ops` | `subquadratic_ops_torch.causal_conv1d` | subquadratic-ops-torch-cu13 0.3.0 |
| `torch_conv1d` | `torch.nn.functional.conv1d` with `groups=dim`, then SiLU | PyTorch 2.14.0+cu130 |
| `cute`, `cute_pad`, `cute_ring` | this repository: ordered, padded and ring conv state | nvidia-cutlass-dsl 4.7.1, apache-tvm-ffi 0.1.14 |

## How each one is called

- **dao.** `causal_conv1d_fwd(x, weight, bias, None, None, out, None, silu)` and
  `causal_conv1d_update(x, state, weight, bias, out, silu, None, None)` on the raw extension,
  skipping the Python autograd wrapper. Compiled with the upstream flags, `--use_fast_math`
  included. That flag matters: without it SiLU prefill is 1.30x slower at the median. Widths 2 to 4
  only. The kernel picks the channel-last path from the strides of `x`.
- **fla_triton.** `causal_conv1d_fwd` on `(batch, seqlen, dim)` input, and
  `causal_conv1d_update(x=, cache=, residual=None, weight=, bias=, activation=)`. FLA keeps a cache
  of the last `width` inputs, shape `(batch, dim, width)`, instead of a `width - 1` state; the
  harness builds that cache from the reference state and checks the trailing `width - 1` columns
  afterwards. Every implementation is called once for the numerical check before it is timed,
  which is when Triton autotunes, so the timed replays use the configuration FLA selected.
- **cudnn.** The public `cudnn.ops` entry points. The op chooses between its own native kernels
  and an engine-graph route; the harness records which route served each case in the `note`
  column. Native kernels exist for width 4 only. With the standard 3-element state the update op
  takes a slower path, so `cudnn_st4` also times it with the 4-element state that its fast path
  needs. `cudnn_st4` is compared against `cute_pad` and `cute_ring`, which also change the state
  layout, and never against the drop-in `cute` kernels.
- **subq_ops.** `causal_conv1d(x, weight, bias, activation)`; prefill only.
- **torch_conv1d.** The reference a user has without any extension. It accumulates in the
  activation dtype, so it is timed but flagged `lowprec`.

Baselines that fail the numerical check are reported as `WRONG` and not timed.

## Timing protocol

`harness.time_stats` captures one call in a CUDA graph and replays it under
`flashinfer.testing.bench_gpu_time_with_cupti`: 10 warm-up replays and 40 timed replays per round,
each timed from CUPTI kernel activity timestamps on the GPU, so host launch cost is excluded. A
round reports the median of its 40 replays. The tables report the median of 5 rounds together with
the fastest and slowest round.

For the cold-cache columns a buffer twice the size of the L2 cache is zeroed and the device is
synchronized before every replay. Both happen before the timed window opens, so the flush is not
part of any reported time. The warm-cache column omits the flush and uses 3 rounds.

`copy_baseline.py` times a device-to-device copy of the same number of bytes with the same
protocol. It is a reference point for how far a kernel is from pure data movement, not a bound.
