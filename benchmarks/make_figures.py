"""Render the README figures from the measured results in benchmarks/results.

Writes five PNG files to docs/figures with Vega-Lite: prefill latency relative to a same-size
device-to-device copy for both tensor layouts, single-token decode latency against batch size with
a cold and with a warm L2 cache, and the three conv paths of a live sglang server.
"""

import csv
from collections import defaultdict
from pathlib import Path
import vl_convert as vlc

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "figures"
FONT = "Latin Modern Roman"
INK, GRID, MUTED = ("#1a1a1a", "#d9d9d9", "#4d4d4d")
OURS = "causal-conv1d-cute"
IMPL = [OURS, "causal-conv1d", "FLA", "cuDNN"]
SPLIT = "split(datum.label, '|')"
COLORS = ["#0072B2", "#009E73", "#CC79A7", "#D55E00"]
SHAPES = ["circle", "triangle-up", "diamond", "square"]
PROVIDER = {
    "cute": OURS,
    "dao": "causal-conv1d",
    "fla_triton": "FLA",
    "cudnn": "cuDNN",
    "cudnn_nwh": "cuDNN",
}
MODEL = {
    "lfm2-350m": "LFM2-350M|(D = 1024, K = 3)",
    "lfm2-1.2b": "LFM2-1.2B|(D = 2048, K = 3)",
    "kimi-k3-kda-tp8": "Kimi-K3, TP8|(D = 1536)",
    "glm5.3-flash-kda-tp4": "GLM-5.3-Flash, TP4|(D = 2048)",
    "qwen3.8-2.4t-gdn-tp8": "Qwen3.8-2.4T, TP8|(D = 2560)",
    "nemotron3-super-tp4": "Nemotron-3 Super, TP4|(D = 2560)",
    "nemotron3-nano-30b": "Nemotron-3 Nano|(D = 6144)",
    "qwen3.5-9b-gdn": "Qwen3.5-9B|(D = 8192)",
    "qwen3.8-27b-gdn": "Qwen3.8-27B|(D = 10240)",
}
DECODE_ONLY = {
    "kimi-k3-kda-tp8-fused": "Kimi-K3 Fused QKV, TP8|(D = 4608)",
    "glm5.3-flash-kda-tp4-fused": "GLM-5.3-Flash Fused QKV, TP4|(D = 6144)",
}
CONFIG = {
    "font": FONT,
    "background": "white",
    "view": {"stroke": None},
    "axis": {
        "labelFont": FONT,
        "titleFont": FONT,
        "labelFontSize": 13,
        "titleFontSize": 14,
        "titleFontWeight": "normal",
        "labelColor": INK,
        "titleColor": INK,
        "domainColor": INK,
        "tickColor": INK,
        "gridColor": GRID,
        "gridDash": [2, 2],
        "domainWidth": 0.8,
        "tickWidth": 0.8,
        "titlePadding": 8,
        "labelLimit": 400,
    },
    "legend": {
        "labelFont": FONT,
        "titleFont": FONT,
        "labelFontSize": 13,
        "titleFontSize": 13,
        "symbolSize": 90,
        "labelLimit": 400,
        "labelColor": INK,
        "titleColor": INK,
        "titleFontWeight": "normal",
    },
    "header": {
        "labelFont": FONT,
        "titleFont": FONT,
        "labelFontSize": 13,
        "labelColor": INK,
        "labelPadding": 6,
        "labelLimit": 400,
    },
    "title": {
        "font": FONT,
        "fontSize": 17,
        "fontWeight": "normal",
        "color": INK,
        "anchor": "start",
        "subtitleFont": FONT,
        "subtitleFontSize": 13,
        "subtitleColor": MUTED,
        "subtitlePadding": 6,
        "offset": 14,
    },
}
SERIES = {
    "color": {
        "field": "impl",
        "type": "nominal",
        "scale": {"domain": IMPL, "range": COLORS},
        "legend": {"title": None, "orient": "top", "direction": "horizontal", "columns": 4},
    },
    "shape": {
        "field": "impl",
        "type": "nominal",
        "scale": {"domain": IMPL, "range": SHAPES},
        "legend": {"title": None, "orient": "top", "direction": "horizontal", "columns": 4},
    },
}


def latencies(column="us_cold"):
    """Load the median latency of every measured case.

    Args:
        column (str): Timing column, "us_cold" or "us_warm".

    Returns:
        dict: Mapping from (model, kind, layout, batch, seqlen, implementation) to the lowest
            median latency in microseconds among the providers drawn as that implementation.
    """
    best = defaultdict(lambda: float("inf"))
    for r in csv.DictReader(open(RES / "kernels_b300.csv")):
        if r["status"] in ("ok", "lowprec") and r["provider"] in PROVIDER:
            key = (
                r["model"],
                r["kind"],
                r["layout"] if r["kind"] == "fwd" else "-",
                int(r["B"]),
                int(r["L"]),
                PROVIDER[r["provider"]],
            )
            best[key] = min(best[key], float(r[column]))
    return best


def save(spec, name):
    """Render a Vega-Lite specification to a PNG file.

    Args:
        spec (dict): Vega-Lite chart specification dictionary.
        name (str): Output filename stem without extension.
    """
    spec = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "config": CONFIG, **spec}
    (OUT / f"{name}.png").write_bytes(vlc.vegalite_to_png(spec, scale=4))
    print("wrote", name)


def prefill(best, layout, name):
    """Render prefill latency relative to a same-size copy, one row per model.

    Args:
        best (dict): Latencies returned by latencies().
        layout (str): Tensor layout key, either "btd" or "bdl".
        name (str): Output filename stem.
        title (str): Chart title.
    """
    copy = {
        r["model"]: float(r["us_copy"])
        for r in csv.DictReader(open(RES / "copy_baseline_b300.csv"))
        if (r["B"], r["L"]) == ("8", "2048")
    }
    data, notes = ([], [])
    for m, label in MODEL.items():
        vals = {i: best.get((m, "fwd", layout, 8, 2048, i)) for i in IMPL}
        vals = {i: v for i, v in vals.items() if v is not None and v != float("inf")}
        for i, v in vals.items():
            data.append({"model": label, "impl": i, "ratio": v / copy[m], "us": v})
        other = min((v for i, v in vals.items() if i != OURS))
        notes.append(
            {"model": label, "us": f"{vals[OURS]:.1f} µs", "gain": f"{other / vals[OURS]:.2f}×"}
        )
    top = max((d["ratio"] for d in data))
    hi = 0.5 * (int(top / 0.5) + 1)
    rows = {
        "field": "model",
        "type": "nominal",
        "sort": list(MODEL.values()),
        "axis": {
            "title": None,
            "grid": True,
            "ticks": False,
            "domain": False,
            "labelPadding": 10,
            "labelExpr": SPLIT,
        },
    }
    ratio = {
        "type": "quantitative",
        "scale": {"domain": [0.75, hi], "nice": False, "zero": False},
        "axis": {
            "title": "Latency Relative to Copy",
            "format": ".1f",
            "values": [1.0 + 0.5 * i for i in range(int((hi - 1.0) / 0.5) + 1)],
        },
    }
    column = lambda field, x, weight: {
        "data": {"values": notes},
        "mark": {
            "type": "text",
            "align": "right",
            "font": FONT,
            "fontSize": 13,
            "fontWeight": weight,
            "color": INK,
        },
        "encoding": {"x": {"value": x}, "y": rows, "text": {"field": field}},
    }
    spec = {
        "width": 620,
        "height": 44 * len(MODEL),
        "layer": [
            {
                "data": {"values": [{"x": 1.0}]},
                "mark": {"type": "rule", "strokeDash": [5, 4], "color": INK, "strokeWidth": 1},
                "encoding": {"x": {"field": "x", **ratio}},
            },
            {
                "data": {"values": [{"x": 1.0, "t": "copy"}]},
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 5,
                    "dy": -6,
                    "baseline": "bottom",
                    "font": FONT,
                    "fontSize": 12,
                    "color": MUTED,
                },
                "encoding": {
                    "x": {"field": "x", **ratio},
                    "y": {"value": 0},
                    "text": {"field": "t"},
                },
            },
            {
                "data": {"values": data},
                "mark": {"type": "point", "filled": True, "size": 110, "opacity": 0.95},
                "encoding": {"x": {"field": "ratio", **ratio}, "y": rows, **SERIES},
            },
            column("us", 700, "normal"),
            column("gain", 760, "bold"),
        ],
    }
    save(spec, name)


def decode(best, cache, name):
    """Render single-token decode latency against batch size, one panel per model.

    Args:
        best (dict): Latencies returned by latencies().
        cache (str): How the L2 cache was prepared, "cold" or "warm", for the subtitle.
        name (str): Output filename stem.
    """
    labels = {**MODEL, **DECODE_ONLY}
    order = [labels[m] for m in labels]
    data = []
    for m, label in labels.items():
        for B in (1, 8, 64, 256):
            for i in IMPL:
                v = best.get((m, "update", "-", B, 1, i))
                if v is not None and v != float("inf"):
                    data.append({"model": label, "impl": i, "B": B, "us": v})
    notes = [
        {"model": labels[m], "B": 1.15, "us": 44.0, "t": "no cuDNN kernel"}
        for m in labels
        if "K = 3" in labels[m]
    ]
    panel = {
        "width": 215,
        "height": 170,
        "layer": [
            {
                "mark": {"type": "line", "strokeWidth": 1.6, "point": {"size": 60, "filled": True}},
                "encoding": {
                    "x": {
                        "field": "B",
                        "type": "quantitative",
                        "scale": {"type": "log", "base": 2, "domain": [1, 256]},
                        "axis": {
                            "title": "Batch Size",
                            "values": [1, 8, 64, 256],
                            "grid": False,
                        },
                    },
                    "y": {
                        "field": "us",
                        "type": "quantitative",
                        "scale": {"type": "log", "base": 2, "domain": [0.9, 64]},
                        "axis": {"title": "Latency (µs)", "values": [1, 2, 4, 8, 16, 32, 64]},
                    },
                    **SERIES,
                },
            },
            {
                "transform": [{"filter": "datum.t != null"}],
                "mark": {
                    "type": "text",
                    "align": "left",
                    "font": FONT,
                    "fontSize": 12,
                    "color": MUTED,
                },
                "encoding": {
                    "x": {"field": "B", "type": "quantitative"},
                    "y": {"field": "us", "type": "quantitative"},
                    "text": {"field": "t"},
                },
            },
        ],
    }
    spec = {
        "data": {"values": data + notes},
        "facet": {
            "field": "model",
            "type": "nominal",
            "sort": order,
            "header": {"title": None, "labelExpr": SPLIT},
        },
        "columns": 6,
        "spacing": {"row": 34, "column": 26},
        "spec": panel,
        "resolve": {"axis": {"x": "independent", "y": "independent"}},
    }
    save(spec, name)


def decode_layouts(column, cache, name):
    """Render decode speedup over the fastest other library for each conv state layout.

    Args:
        column (str): Timing column, "us_cold" or "us_warm".
        cache (str): How the L2 cache was prepared, "cold" or "warm", for the title.
        name (str): Output filename stem.
    """
    layouts = {
        "cute": "Ordered State",
        "cute_pad": "Padded State",
        "cute_ring": "Ring State",
    }
    labels = {**MODEL, **DECODE_ONLY}
    times = defaultdict(dict)
    for r in csv.DictReader(open(RES / "kernels_b300.csv")):
        if r["kind"] == "update" and r["status"] == "ok" and r[column] not in ("", "nan"):
            times[r["model"], int(r["B"])][r["provider"]] = float(r[column])
    data = []
    for (m, B), t in times.items():
        other = min((v for p, v in t.items() if p not in layouts and p != "torch_conv1d"))
        for p, label in layouts.items():
            if p in t:
                data.append(
                    {"model": labels[m], "B": f"B = {B}", "layout": label, "gain": other / t[p]}
                )
    hi = max((d["gain"] for d in data))
    gain = {
        "field": "gain",
        "type": "quantitative",
        "scale": {"type": "log", "domain": [0.7, 1.05 * hi], "nice": False},
        "axis": {"title": "Speedup (×)", "values": [0.75, 1, 1.5, 2, 3], "format": "~g"},
    }
    rows = {
        "field": "model",
        "type": "nominal",
        "sort": list(labels.values()),
        "axis": {
            "title": None,
            "grid": True,
            "ticks": False,
            "domain": False,
            "labelPadding": 8,
            "labelExpr": SPLIT,
        },
    }
    panel = {
        "width": 170,
        "height": 40 * len(labels),
        "layer": [
            {
                "mark": {"type": "rule", "strokeDash": [5, 4], "color": INK, "strokeWidth": 1},
                "encoding": {
                    "x": {"datum": 1.0, **{k: v for k, v in gain.items() if k != "field"}}
                },
            },
            {
                "mark": {"type": "point", "filled": True, "size": 80, "opacity": 0.95},
                "encoding": {
                    "x": gain,
                    "y": rows,
                    "color": {
                        "field": "layout",
                        "type": "nominal",
                        "scale": {"domain": list(layouts.values()), "range": COLORS[:3]},
                        "legend": {"title": None, "orient": "top", "direction": "horizontal"},
                    },
                    "shape": {
                        "field": "layout",
                        "type": "nominal",
                        "scale": {"domain": list(layouts.values()), "range": SHAPES[:3]},
                    },
                },
            },
        ],
    }
    spec = {
        "data": {"values": data},
        "facet": {
            "column": {
                "field": "B",
                "type": "nominal",
                "sort": ["B = 1", "B = 8", "B = 64", "B = 256"],
                "header": {"title": None},
            }
        },
        "spacing": {"column": 18},
        "spec": panel,
    }
    save(spec, name)


def sglang(model="qwen3.8-27b", name="sglang_paths"):
    """Render the three conv paths of a live sglang server: prefill, verify and decode.

    Args:
        model (str): Model name in sglang_paths_b300.csv.
        name (str): Output filename stem.
    """
    rows = [r for r in csv.DictReader(open(RES / "sglang_paths_b300.csv")) if r["model"] == model]
    names = ["SGLang", OURS]
    panels = [
        (
            "prefill",
            "Prefill",
            "Tokens",
            lambda r: r["seqs"] == "1" and int(r["tokens"]) >= 64,
        ),
        ("verify", "MTP Verify", "Batch Size", lambda r: True),
        ("decode", "Decode", "Batch Size", lambda r: True),
    ]
    charts = []
    for path, title, xlabel, keep in panels:
        sel = [r for r in rows if r["path"] == path and keep(r)]
        xs = [int(r["tokens"] if path == "prefill" else r["seqs"]) for r in sel]
        data, notes = ([], [])
        for r, x in zip(sel, xs):
            a, b = (float(r["us_stock"]), float(r["us_cute"]))
            data += [{"x": x, "us": a, "impl": names[0]}, {"x": x, "us": b, "impl": names[1]}]
            text = f"{a / b:.2f}×"
            notes.append(
                {
                    "x": x,
                    "us": max(a, b),
                    "t": text,
                    "first": x == xs[0],
                    "show": (len(notes) % 2 == len(sel) % 2 - 1 or len(notes) == len(sel) - 1)
                    and abs(a / b - 1) >= 0.015,
                }
            )
        lo, hi = (min((d["us"] for d in data)), max((d["us"] for d in data)))
        ticks = [v for v in (2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128) if 0.8 * lo <= v <= 1.5 * hi]
        label = lambda first: {
            "data": {"values": [n for n in notes if n["first"] == first and n["show"]]},
            "mark": {
                "type": "text",
                "dy": -10,
                "dx": 4 if first else 0,
                "align": "left" if first else "center",
                "font": FONT,
                "fontSize": 11.5,
                "color": MUTED,
            },
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "us", "type": "quantitative"},
                "text": {"field": "t"},
            },
        }
        charts.append(
            {
                "title": {"text": title, "fontSize": 14, "anchor": "middle", "offset": 8},
                "width": 270,
                "height": 220,
                "layer": [
                    {
                        "data": {"values": data},
                        "mark": {
                            "type": "line",
                            "strokeWidth": 1.6,
                            "point": {"size": 60, "filled": True},
                        },
                        "encoding": {
                            "x": {
                                "field": "x",
                                "type": "quantitative",
                                "scale": {"type": "log", "base": 2},
                                "axis": {
                                    "title": xlabel,
                                    "values": xs[::2] if len(xs) > 6 else xs,
                                    "grid": False,
                                    "labelOverlap": False,
                                    "format": "d",
                                },
                            },
                            "y": {
                                "field": "us",
                                "type": "quantitative",
                                "scale": {
                                    "type": "log",
                                    "domain": [0.8 * lo, 1.5 * hi],
                                    "nice": False,
                                },
                                "axis": {"title": "Latency (µs)", "values": ticks},
                            },
                            "color": {
                                "field": "impl",
                                "type": "nominal",
                                "scale": {"domain": names, "range": ["#D55E00", "#0072B2"]},
                                "legend": {
                                    "title": None,
                                    "orient": "top",
                                    "direction": "horizontal",
                                },
                            },
                            "shape": {
                                "field": "impl",
                                "type": "nominal",
                                "scale": {"domain": names, "range": ["square", "circle"]},
                                "legend": {
                                    "title": None,
                                    "orient": "top",
                                    "direction": "horizontal",
                                },
                            },
                        },
                    },
                    label(True),
                    label(False),
                ],
            }
        )
    spec = {
        "hconcat": charts,
        "spacing": 30,
    }
    save(spec, name)


def row_stride():
    """Render channel-last prefill latency against the distance between token rows."""
    data = []
    for r in csv.DictReader(open(RES / "row_stride_b300.csv")):
        shape = f"D = {r['dim']}, K = {r['width']}"
        spacing = "Power of Two" if r["power_of_two"] == "1" else "Other"
        for col, kernel in (
            ("us_single", "1 Strip per Thread"),
            ("us_macro", "6 Strips per Thread"),
        ):
            data.append(
                {
                    "shape": shape,
                    "rows": f"{int(r['row_elements']):,}",
                    "order": int(r["row_elements"]),
                    "kernel": kernel,
                    "spacing": spacing,
                    "us": float(r[col]),
                }
            )
    shapes = list(dict.fromkeys((d["shape"] for d in data)))
    spec = {
        "data": {"values": data},
        "facet": {"field": "shape", "type": "nominal", "sort": shapes, "header": {"title": None}},
        "columns": 4,
        "spacing": {"column": 30},
        "spec": {
            "width": 230,
            "height": 200,
            "mark": {"type": "point", "filled": True, "size": 90},
            "encoding": {
                "x": {
                    "field": "rows",
                    "type": "nominal",
                    "sort": {"field": "order"},
                    "axis": {"title": "Row Stride (elements)", "labelAngle": -35},
                },
                "y": {
                    "field": "us",
                    "type": "quantitative",
                    "scale": {"zero": False},
                    "axis": {"title": "Latency (µs)", "tickCount": 5},
                },
                "color": {
                    "field": "kernel",
                    "type": "nominal",
                    "scale": {
                        "domain": ["1 Strip per Thread", "6 Strips per Thread"],
                        "range": ["#D55E00", "#0072B2"],
                    },
                    "legend": {"title": None, "orient": "top", "direction": "horizontal"},
                },
                "shape": {
                    "field": "spacing",
                    "type": "nominal",
                    "scale": {
                        "domain": ["Power of Two", "Other"],
                        "range": ["diamond", "circle"],
                    },
                    "legend": {
                        "title": "Row Stride",
                        "orient": "top",
                        "direction": "horizontal",
                    },
                },
            },
        },
        "resolve": {"scale": {"y": "independent", "x": "independent"}},
    }
    save(spec, "row_stride")


def main():
    """Render every figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    vlc.register_font_directory("/usr/share/texmf/fonts/opentype/public/lm")
    best = latencies()
    prefill(best, "btd", "prefill_channel_last")
    prefill(best, "bdl", "prefill_contiguous")
    decode(best, "cold", "decode_batch_scaling")
    decode(latencies("us_warm"), "warm", "decode_batch_scaling_warm")
    decode_layouts("us_warm", "warm", "decode_state_layouts_warm")
    decode_layouts("us_cold", "cold", "decode_state_layouts")
    sglang()
    if (RES / "row_stride_b300.csv").exists():
        row_stride()


if __name__ == "__main__":
    main()
