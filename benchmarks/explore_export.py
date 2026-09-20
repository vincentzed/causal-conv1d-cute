"""Flatten the JSON lines of explore_decode.py into one CSV row per configuration.

python explore_export.py explore results/explore_decode_qwen3.8-27b_b300.csv
"""

import csv
import json
import sys
from pathlib import Path


def main():
    """Write the CSV."""
    root, out = (Path(sys.argv[1]), Path(sys.argv[2]))
    recs = []
    for f in sorted(root.glob("shard_*.jsonl")):
        recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    recs.sort(key=lambda r: r["id"])
    batches = sorted({B for r in recs for B in r.get("us", {})}, key=int)
    names = list(recs[0]["params"])
    ptx = ["lines", "ld_global", "st_global", "fma_bf16", "fma_f32", "cvt", "div"]
    head = ["id"] + [n for n in names if n != "thread"] + ["tiles", "hoist", "guard", "correct"]
    head += ["compile_s"] + [f"ptx_{k}" for k in ptx]
    head += [f"us_{c}_b{B}" for B in batches for c in ("warm", "cold")]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in recs:
            p = r["params"]
            row = [r["id"]] + [p[n] for n in names if n != "thread"]
            row += [p["thread"][0], int(p["thread"][1]), int(r.get("guard", False))]
            row += [int(r.get("correct", False)), r.get("compile_s", "")]
            row += [r.get("ptx", {}).get(k, "") for k in ptx]
            row += [
                r.get("us", {}).get(B, {}).get(c, "") for B in batches for c in ("warm", "cold")
            ]
            w.writerow(row)
    print(f"wrote {len(recs)} configurations to {out}")


if __name__ == "__main__":
    main()
