"""Summarize a benchmark CSV as wins, ties and losses against the best other implementation.

Two comparisons are reported. The drop-in kernels keep the standard (batch, dim, width - 1) conv
state and are compared against every implementation that does the same. The padded-state decode
kernels and the ring-state decode kernels change the state layout, as cuDNN's native width-4 state
kernel does, and are compared against every implementation including that one.

Each comparison is made twice, once on the cold-cache timings and once on the warm-cache timings.
A cold-cache case is a win when the slowest of our rounds is faster than the fastest round of the
best other implementation, a loss in the reverse situation, and a tie when the ranges overlap.
Warm-cache timings carry no range, so a case within 2% either way is a tie.
"""

import collections
import csv
import sys

TIE = 0.02
MINE = ("cute", "cute_pad", "cute_ring")
EXCL = MINE + ("torch_conv1d",)
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    if r["status"] in ("WRONG", "ERROR"):
        print(
            f"!! {r['provider']} {r['status']} {r['model']} {r['kind']} B{r['B']} L{r['L']} {r['layout']}: {r['note'][:90]}"
        )
flag = collections.Counter((r["provider"] for r in rows if r["status"] == "lowprec"))
print("lowprec rows (timed, flagged):", dict(flag) or "none")
cases = collections.OrderedDict()
for r in rows:
    if r["status"] in ("ok", "lowprec"):
        k = (
            r["model"],
            r["kind"],
            r["layout"] if r["kind"] == "fwd" else "-",
            int(r["B"]),
            int(r["L"]),
            int(r["W"]),
        )
        cases.setdefault(k, {})[r["provider"]] = {
            c: float(r[c]) if r.get(c) not in (None, "", "nan") else float("nan")
            for c in ("us_cold", "us_cold_lo", "us_cold_hi", "us_warm")
        }


def regime(k):
    """Classify a benchmark case into an execution regime.

    Args:
        k (tuple): Case description tuple consisting of (model, kind, layout, batch,
            sequence_length, kernel_width).

    Returns:
        str: Execution regime string identifying prefill memory layout or
        decode batch size range.
    """
    if k[1] == "fwd":
        return f"prefill {('channel-last' if k[2] == 'btd' else 'contiguous  ')}"
    return "decode B<=8          " if k[3] <= 8 else "decode B>=64         "


def outcome(me, other, col):
    """Classify one case as a win, tie or loss.

    Args:
        me (dict): Timings of our kernel.
        other (dict): Timings of the best other implementation.
        col (str): Timing column, either "us_cold" or "us_warm".

    Returns:
        str: One of "win", "tie" or "loss".
    """
    if col == "us_cold":
        if me["us_cold_hi"] < other["us_cold_lo"]:
            return "win"
        return "loss" if other["us_cold_hi"] < me["us_cold_lo"] else "tie"
    ratio = other[col] / me[col]
    return "win" if ratio > 1 + TIE else "loss" if ratio < 1 - TIE else "tie"


for col, title in (("us_cold", "COLD L2 (flushed before every replay)"), ("us_warm", "WARM L2")):
    for mine, note in (
        ("cute", "drop-in, standard conv state"),
        ("cute_pad", "padded conv state, decode only, vs everyone incl. cudnn_st4"),
        ("cute_ring", "ring conv state, decode only, vs everyone incl. cudnn_st4"),
    ):
        print(f"\n================ {title} | {mine}: {note} ================")
        agg = collections.OrderedDict()
        losses = []
        for k, d in cases.items():
            if mine not in d or (mine != "cute" and k[1] != "update"):
                continue
            others = {
                p: v
                for p, v in d.items()
                if p not in EXCL and (mine != "cute" or p != "cudnn_st4") and v[col] == v[col]
            }
            if not others or d[mine][col] != d[mine][col]:
                continue
            who = min(others, key=lambda p: others[p][col])
            ratio = others[who][col] / d[mine][col]
            res = outcome(d[mine], others[who], col)
            agg.setdefault(regime(k), []).append((ratio, res))
            if res == "loss":
                losses.append((ratio, k, d[mine][col], who, others[who][col]))
        n = collections.Counter((res for v in agg.values() for _, res in v))
        for reg, v in agg.items():
            c = collections.Counter((res for _, res in v))
            rs = sorted((x for x, _ in v))
            print(
                f"  {reg} n={len(v):<3} win {c['win']:>3}  tie {c['tie']:>3}  loss {c['loss']:>3}   speedup vs best other: min {rs[0]:.2f}x  median {rs[len(rs) // 2]:.2f}x  max {rs[-1]:.2f}x"
            )
        print(f"  TOTAL: win {n['win']}  tie {n['tie']}  loss {n['loss']}  of {sum(n.values())}")
        for ratio, k, m, who, o in sorted(losses):
            print(
                f"    LOSS {ratio:.2f}x  {k[0]} {k[1]} {k[2]} B{k[3]} L{k[4]}: ours {m:.2f} vs {who} {o:.2f}"
            )
