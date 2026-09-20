"""Render the design-space figure from the CSV written by explore_export.py.

One panel per batch size shows how the warm-cache latency of every explored decode kernel
configuration is distributed, with the latency of each other implementation on the same shape
marked on the same axis.

    python make_explore_figure.py results/explore_decode_qwen3.8-27b_b300.csv
"""

import csv
import sys
from make_figures import INK, OUT, RES, save
import vl_convert as vlc

OTHERS = {
    "dao": "causal-conv1d",
    "fla_triton": "FLA",
    "cudnn": "cuDNN",
    "cudnn_st4": "cuDNN (Padded State)",
}
COLORS = {
    "causal-conv1d": "#009E73",
    "FLA": "#CC79A7",
    "cuDNN": "#D55E00",
    "cuDNN (Padded State)": "#E69F00",
}


def main():
    """Render the figure."""
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["correct"] == "1"]
    model = "qwen3.8-27b-gdn"
    others, ring = ({}, {})
    for r in csv.DictReader(open(RES / "kernels_b300.csv")):
        if r["model"] == model and r["kind"] == "update" and r["provider"] in OTHERS:
            others[int(r["B"]), OTHERS[r["provider"]]] = float(r["us_warm"])
        if r["model"] == model and r["provider"] == "cute_ring":
            ring[int(r["B"])] = float(r["us_warm"])
    panels = []
    for B in (1, 8, 64, 256):
        col = f"us_warm_b{B}"
        data = [
            {
                "us": float(r[col]),
                "layout": "standard state" if r["layout"] == "std" else "4-element state",
            }
            for r in rows
            if r.get(col)
        ]
        best = {
            k: min((d["us"] for d in data if d["layout"] == k))
            for k in ("standard state", "4-element state")
        }
        marks = [{"us": v, "who": who} for (b, who), v in others.items() if b == B]
        lo = min(min(best.values()), ring[B])
        hi = max([m["us"] for m in marks] + [sorted((d["us"] for d in data))[int(0.9 * len(data))]])
        panels.append(
            {
                "title": {
                    "text": f"B = {B}",
                    "fontSize": 14,
                    "anchor": "middle",
                    "offset": 8,
                },
                "width": 300,
                "height": 170,
                "layer": [
                    {
                        "data": {"values": [d for d in data if d["us"] <= 1.05 * hi]},
                        "mark": {
                            "type": "bar",
                            "color": "#0072B2",
                            "opacity": 0.55,
                            "binSpacing": 0,
                        },
                        "encoding": {
                            "x": {
                                "field": "us",
                                "type": "quantitative",
                                "bin": {"maxbins": 50},
                                "scale": {"domain": [0.95 * lo, 1.05 * hi], "nice": False},
                                "axis": {
                                    "title": "Latency (µs)",
                                    "grid": False,
                                    "tickCount": 6,
                                },
                            },
                            "y": {
                                "aggregate": "count",
                                "type": "quantitative",
                                "axis": {"title": "Configurations", "tickCount": 4},
                            },
                        },
                    },
                    {
                        "data": {"values": marks},
                        "mark": {"type": "rule", "strokeWidth": 2},
                        "encoding": {
                            "x": {"field": "us", "type": "quantitative"},
                            "color": {
                                "field": "who",
                                "type": "nominal",
                                "scale": {"domain": list(COLORS), "range": list(COLORS.values())},
                                "legend": {
                                    "title": None,
                                    "orient": "top",
                                    "direction": "horizontal",
                                },
                            },
                        },
                    },
                    {
                        "data": {"values": [{"us": best["standard state"]}]},
                        "mark": {
                            "type": "rule",
                            "strokeWidth": 2,
                            "strokeDash": [6, 3],
                            "color": INK,
                        },
                        "encoding": {"x": {"field": "us", "type": "quantitative"}},
                    },
                    {
                        "data": {"values": [{"us": ring[B]}]},
                        "mark": {"type": "rule", "strokeWidth": 2, "color": INK},
                        "encoding": {"x": {"field": "us", "type": "quantitative"}},
                    },
                    {
                        "data": {"values": [{"us": best["4-element state"]}]},
                        "mark": {
                            "type": "rule",
                            "strokeWidth": 2,
                            "strokeDash": [1.5, 3],
                            "color": INK,
                        },
                        "encoding": {"x": {"field": "us", "type": "quantitative"}},
                    },
                ],
            }
        )
    spec = {
        "hconcat": panels,
        "spacing": 36,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    vlc.register_font_directory("/usr/share/texmf/fonts/opentype/public/lm")
    save(spec, "explore_decode")


if __name__ == "__main__":
    main()
