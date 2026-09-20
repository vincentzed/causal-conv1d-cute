"""Benchmark causal convolution implementations across model shapes.

Executes kernels with cold L2 cache flushing and optional warm L2 cache timing using CUDA graph
replays. Numerical checks evaluate correctness against an FP32 reference implementation:
- ok: matches FP32 accumulation reference behavior.
- lowprec: uses reduced-precision accumulation.
- WRONG: invalid numerical output or incorrect conv state shift. Results including latencies,
  numerical error, and memory throughput are written to a CSV file.
"""

import argparse
import csv
from pathlib import Path
import torch
import harness as H
import providers as P
import shapes as S

ALL = ["dao", "fla_triton", "cudnn", "cudnn_nwh", "subq_ops", "torch_conv1d"]


@torch.no_grad()
def main():
    """Run benchmarks across shapes and kernel implementations.

    Parses command-line arguments, executes forward and update operations across requested tensor
    shapes and kernel implementations, verifies outputs and conv state updates against a reference
    implementation, measures cold and warm L2 cache latencies using CUDA graph replay, and writes
    benchmark metrics to a CSV file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--providers", nargs="*", default=ALL)
    ap.add_argument("--layouts", nargs="*", default=["btd", "bdl"])
    ap.add_argument("--kinds", nargs="*", default=["fwd", "update"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument(
        "--warm", action="store_true", help="also time with a warm L2 (cold is the headline)"
    )
    a = ap.parse_args()
    provs, failed = P.load(a.providers)
    for n, why in failed.items():
        print(f"!! provider {n} failed to load: {why}", flush=True)
    cases = S.cases(a.models, a.layouts, a.kinds)
    print(
        f"{len(cases)} cases x {len(provs)} providers on {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "model",
                "kind",
                "D",
                "W",
                "B",
                "L",
                "layout",
                "bias",
                "act",
                "provider",
                "status",
                "ulp_err",
                "bwd_err",
                "us_cold",
                "us_cold_lo",
                "us_cold_hi",
                "us_warm",
                "gb_per_s_cold",
                "note",
            ]
        )
        for case in cases:
            t = H.make_inputs(case)
            ref = H.reference(case, t)
            print(f"\n## {case.label}", flush=True)
            for p in provs:
                p.skip_reason, p.note = ("", "")
                row = [
                    case.model,
                    case.kind,
                    case.D,
                    case.W,
                    case.B,
                    case.L,
                    case.layout,
                    int(case.bias),
                    case.act or "none",
                    p.name,
                ]
                try:
                    made = p.fwd(case, t) if case.kind == "fwd" else p.update(case, t)
                    if made is None:
                        wr.writerow(row + ["skip"] + [""] * 7 + [p.skip_reason])
                        f.flush()
                        print(f"   {p.name:<13} skip: {p.skip_reason}", flush=True)
                        continue
                    fn, get_out = (made[0], made[1])
                    fn()
                    torch.cuda.synchronize()
                    status, err, bwd = H.check(
                        case, t, get_out(), ref if case.kind == "fwd" else ref[0]
                    )
                    if case.kind == "update" and (not torch.equal(made[2](), ref[1])):
                        status, p.note = ("WRONG", "conv_state not shifted correctly")
                    if status == "WRONG":
                        wr.writerow(
                            row + ["WRONG", f"{err:.3g}", f"{bwd:.3g}"] + [""] * 5 + [p.note]
                        )
                        f.flush()
                        print(
                            f"   {p.name:<13} WRONG: bwd_err {bwd:.3g}, {err:.3g} ulp -- not timed {p.note}",
                            flush=True,
                        )
                        continue
                    stats = H.time_stats(fn, rounds=a.rounds, cold=True, graph=p.graph)
                    cold = stats["med"]
                    warm = (
                        H.time_us(fn, rounds=3, cold=False, graph=p.graph)
                        if a.warm
                        else float("nan")
                    )
                    gbs = case.bytes_moved / (cold * 1e-06) / 1000000000.0
                    wr.writerow(
                        row
                        + [
                            status,
                            f"{err:.3g}",
                            f"{bwd:.3g}",
                            f"{cold:.3f}",
                            f"{stats['lo']:.3f}",
                            f"{stats['hi']:.3f}",
                            f"{warm:.3f}",
                            f"{gbs:.1f}",
                            p.note,
                        ]
                    )
                    f.flush()
                    flag = "" if status == "ok" else f"  [LOWPREC bwd_err {bwd:.2g}]"
                    print(
                        f"   {p.name:<13} {cold:10.2f} us  ({gbs:8.1f} GB/s, err {err:.2f} ulp){flag} {p.note}",
                        flush=True,
                    )
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e)[:160]}"
                    wr.writerow(row + ["ERROR"] + [""] * 7 + [msg])
                    f.flush()
                    print(f"   {p.name:<13} ERROR {msg}", flush=True)
                    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
