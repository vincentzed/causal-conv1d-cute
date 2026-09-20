"""Measure how the distance between token rows affects channel-last prefill.

A serving engine often passes the conv a view of a wider buffer, for example the leading columns
of a fused projection output, so consecutive tokens are further apart than dim elements. When that
distance is a power of two in bytes, the width - 1 tokens that a strip shares with its predecessor
map to the same cache sets as the strip's own rows and are evicted before they are read again.

Each case below runs the same convolution on the same number of tokens and changes only the
distance between rows. It is timed twice: with one strip per thread, which loads the shared tokens
once per strip, and with six strips per thread, which keeps them in registers between strips.

    python row_stride.py results/row_stride_b300.csv
"""

import csv
import sys
import torch
import harness as H
from causal_conv1d_cute.api import CausalConv1d

CASES = [(2048, 4), (2048, 3), (8192, 4), (10240, 4)]
TOKENS = 8192


@torch.no_grad()
def main():
    """Write one row per (dim, width, row distance)."""
    torch.manual_seed(0)
    with open(sys.argv[1], "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["dim", "width", "row_elements", "row_bytes", "power_of_two", "us_single", "us_macro"]
        )
        for D, W in CASES:
            w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16) * 0.3
            layer = CausalConv1d(w, None, "silu")
            out = torch.empty(TOKENS, D, device="cuda", dtype=torch.bfloat16).t().unsqueeze(0)
            pow2 = 1 << (D - 1).bit_length()
            for cols in sorted({D, D + 16, D + 1024, pow2, 2 * pow2}):
                x = torch.randn(TOKENS, cols, device="cuda", dtype=torch.bfloat16)[:, :D]
                x = x.t().unsqueeze(0)
                single = H.time_us(lambda: layer.fwd(x, out=out, cfg=("strip", 5, 64)))
                macro = H.time_us(lambda: layer.fwd(x, out=out, cfg=("strip", 3, 64, 6)))
                row = [
                    D,
                    W,
                    cols,
                    2 * cols,
                    int(cols & (cols - 1) == 0),
                    f"{single:.3f}",
                    f"{macro:.3f}",
                ]
                wr.writerow(row)
                f.flush()
                print(*row, flush=True)


if __name__ == "__main__":
    main()
