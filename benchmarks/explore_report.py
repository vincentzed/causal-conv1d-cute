"""Summarize the JSON lines written by explore_decode.py.

Prints, for every batch size and cache state, the best configurations of each conv state layout,
the main effect of every parameter (the geometric mean latency of the configurations that share a
value, relative to the best value), and the strongest pairwise interactions.

    python explore_report.py explore [--top 8]
"""

import argparse
import collections
import itertools
import json
import math
from pathlib import Path


def load(root):
    """Read every shard and return the records that compiled and passed the reference check."""
    recs = []
    for f in sorted(Path(root).glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    good = [r for r in recs if r.get("correct") and "us" in r]
    for r in good:
        r["params"]["thread"] = tuple(r["params"]["thread"])
    return (recs, good)


def gmean(values):
    """Return the geometric mean of a non-empty sequence."""
    values = list(values)
    return math.exp(sum((math.log(v) for v in values)) / len(values))


def main():
    """Print the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    recs, good = load(a.root)
    bad = [r for r in recs if "error" in r]
    wrong = [r for r in recs if r.get("correct") is False]
    print(
        f"{len(recs)} configurations: {len(good)} timed, {len(wrong)} failed the check, {len(bad)} errors"
    )
    for e, n in collections.Counter((r["error"][:90] for r in bad)).most_common(5):
        print(f"   {n:5d}  {e}")
    names = list(good[0]["params"])
    batches = sorted({B for r in good for B in r["us"]}, key=int)
    everything = good
    for B in batches:
        good = [r for r in everything if B in r["us"]]
        for cache in ("warm", "cold"):
            t = lambda r: r["us"][B][cache]
            print(f"\n================ B = {B}, {cache} L2 ================")
            for layout in ("std", "pad"):
                rows = sorted((r for r in good if r["params"]["layout"] == layout), key=t)
                print(
                    f"  {layout}: best {t(rows[0]):.3f} us, median {t(rows[len(rows) // 2]):.3f}, worst {t(rows[-1]):.3f}"
                )
                for r in rows[: a.top]:
                    p = r["params"]
                    other = "cold" if cache == "warm" else "warm"
                    print(
                        f"     {t(r):7.3f} ({other} {r['us'][B][other]:.3f})  cv={p['cv']:<2} tiles={p['thread'][0]} hoist={int(p['thread'][1])} rows={p.get('rows', 1)} bs={p['bs']:<3} {p['grid']} w={p['wlayout']} {p['acc']:<5} loads={p['loads']} stores={p['stores']} mbp={p['mbp']}  ld/st={r['ptx'].get('ld_global')}/{r['ptx'].get('st_global')}"
                    )
            print(
                "  main effects (geometric mean latency relative to the best value of each parameter):"
            )
            for name in names:
                groups = collections.defaultdict(list)
                for r in good:
                    groups[r["params"][name]].append(t(r))
                means = {k: gmean(v) for k, v in groups.items()}
                lo = min(means.values())
                print(
                    f"     {name:<8} "
                    + "  ".join(
                        f"{k}={m / lo:.3f}" for k, m in sorted(means.items(), key=lambda kv: kv[1])
                    )
                )
    print("\n================ strongest pairwise interactions (B = 64, warm) ================")
    good = [r for r in everything if "64" in r["us"]]
    t = lambda r: math.log(r["us"]["64"]["warm"])
    if good:
        mu = sum((t(r) for r in good)) / len(good)
        main = {n: collections.defaultdict(list) for n in names}
        for r in good:
            for n in names:
                main[n][r["params"][n]].append(t(r))
        eff = {n: {k: sum(v) / len(v) - mu for k, v in d.items()} for n, d in main.items()}
        scores = []
        for n1, n2 in itertools.combinations(names, 2):
            cells = collections.defaultdict(list)
            for r in good:
                cells[r["params"][n1], r["params"][n2]].append(t(r))
            inter = {
                k: sum(v) / len(v) - mu - eff[n1][k[0]] - eff[n2][k[1]] for k, v in cells.items()
            }
            k = max(inter, key=lambda c: abs(inter[c]))
            scores.append((abs(inter[k]), n1, n2, k, inter[k]))
        for s, n1, n2, k, v in sorted(scores, reverse=True)[:8]:
            print(
                f"     {n1} x {n2}: largest deviation {math.exp(v):.3f}x at {n1}={k[0]}, {n2}={k[1]}"
            )


if __name__ == "__main__":
    main()
