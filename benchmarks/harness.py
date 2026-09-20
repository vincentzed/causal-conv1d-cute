"""Provide benchmark utilities and test cases for convolution kernels.

Timing measurements use CUPTI under CUDA graph replay with optional L2 cache flushing. Numerical
validation compares kernel outputs against an unrounded fp32 reference implementation.
"""

import statistics
from flashinfer.testing.utils import bench_gpu_time_with_cupti
from causal_conv1d_cute.testing import (
    MANT_BITS,
    REF_MIN,
    Case,
    check,
    l1_magnitude,
    make_inputs,
    reference,
)

__all__ = [
    "MANT_BITS",
    "REF_MIN",
    "Case",
    "check",
    "l1_magnitude",
    "make_inputs",
    "reference",
    "time_stats",
    "time_us",
]


def time_stats(fn, rounds=5, cold=True, dry=10, rep=40, graph=True):
    """Time fn with CUPTI and return the spread across rounds.

    Each round runs dry warm-up replays and rep timed replays of one captured CUDA graph and keeps
    the median GPU kernel span. With cold=True a buffer twice the size of L2 is zeroed and the
    device synchronized before every replay, outside the timed window.

    Returns:
      Dict with the median, minimum and maximum of the per-round medians, in microseconds.
    """
    meds = []
    for _ in range(rounds):
        t = bench_gpu_time_with_cupti(
            fn, dry_run_iters=dry, repeat_iters=rep, use_cuda_graph=graph, cold_l2_cache=cold
        )
        meds.append(statistics.median(t) * 1e3)
    return {"med": statistics.median(meds), "lo": min(meds), "hi": max(meds)}


def time_us(fn, rounds=3, cold=True, dry=10, rep=40, graph=True):
    """Return the median over rounds of the per-round median kernel time in microseconds."""
    return time_stats(fn, rounds, cold, dry, rep, graph)["med"]
