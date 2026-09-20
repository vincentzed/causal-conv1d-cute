"""Render the end-to-end figures for the sglang integration from benchmarks/results.

Writes four PNG files to docs/figures: the breakdown of GPU kernel time inside a running server,
the conv kernel time per call inside that server, the serving benchmarks of every session relative
to the stock mean, and the numerical agreement with a float64 reference together with GSM8K.
"""

import csv
from collections import defaultdict
from make_figures import FONT, INK, OUT, RES, save
import vl_convert as vlc

STOCK, PLUGIN = ("SGLang", "causal-conv1d-cute")
SPLIT = "split(datum.label, '|')"
ARM = {"stock": STOCK, "plugin": PLUGIN}
PAIR = {"domain": [STOCK, PLUGIN], "range": ["#D55E00", "#0072B2"]}
GROUPS = {
    "GEMM": "GEMM",
    "delta-rule recurrence": "Delta Rule",
    "normalization": "Normalization",
    "conv": "Conv1d",
}
PARTS = ["GEMM", "Delta Rule", "Normalization", "Other", "Conv1d"]
PART_COLORS = ["#c9c9c9", "#009E73", "#CC79A7", "#e6e6e6", "#0072B2"]


def rows(name):
    """Read one results file as a list of dictionaries."""
    return list(csv.DictReader(open(RES / name)))


def breakdown():
    """Render the share of GPU kernel time by kernel family, stock and plugin, prefill and decode."""
    share = defaultdict(list)
    for r in rows("sglang_profile_b300.csv"):
        part = GROUPS.get(r["category"], "Other")
        share[r["window"], r["arm"], r["session"], part].append(float(r["share_pct"]))
    total = defaultdict(float)
    for (window, arm, session, part), v in share.items():
        total[window, arm, part] += sum(v) / 2
    label = {
        ("prefill", "stock"): "Prefill|(SGLang)",
        ("prefill", "plugin"): "Prefill|(causal-conv1d-cute)",
        ("decode", "stock"): "MTP Decode|(SGLang)",
        ("decode", "plugin"): "MTP Decode|(causal-conv1d-cute)",
    }
    data, notes = ([], [])
    for (window, arm), name in label.items():
        for i, part in enumerate(PARTS):
            data.append({"row": name, "part": part, "order": i, "share": total[window, arm, part]})
        notes.append({"row": name, "t": f"conv1d {total[window, arm, 'Conv1d']:.2f}%"})
    order = list(label.values())
    spec = {
        "width": 720,
        "height": 40 * len(order),
        "layer": [
            {
                "data": {"values": data},
                "mark": {"type": "bar", "height": 22, "stroke": "white", "strokeWidth": 0.6},
                "encoding": {
                    "x": {
                        "field": "share",
                        "type": "quantitative",
                        "stack": "zero",
                        "scale": {"domain": [0, 100]},
                        "axis": {
                            "title": "GPU Time (%)",
                            "grid": False,
                            "tickCount": 5,
                        },
                    },
                    "y": {
                        "field": "row",
                        "type": "nominal",
                        "sort": order,
                        "axis": {
                            "title": None,
                            "ticks": False,
                            "domain": False,
                            "labelPadding": 8,
                            "labelExpr": SPLIT,
                        },
                    },
                    "color": {
                        "field": "part",
                        "type": "nominal",
                        "scale": {"domain": PARTS, "range": PART_COLORS},
                        "legend": {"title": None, "orient": "top", "direction": "horizontal"},
                    },
                    "order": {"field": "order", "type": "quantitative"},
                },
            },
            {
                "data": {"values": notes},
                "mark": {
                    "type": "text",
                    "align": "left",
                    "font": FONT,
                    "fontSize": 13,
                    "color": INK,
                },
                "encoding": {
                    "x": {"value": 730},
                    "y": {"field": "row", "type": "nominal", "sort": order},
                    "text": {"field": "t"},
                },
            },
        ],
    }
    save(spec, "e2e_gpu_time_breakdown")


def conv_in_server():
    """Render the mean conv kernel time per call inside the running server."""
    name = {
        "prefill": "Prefill (2048 Tokens)",
        "mtp_verify": "MTP Verify (16 Requests)",
        "decode_c16": "Decode (16 Requests)",
        "decode_c64": "Decode (64 Requests)",
        "fn_decode_c64": "Flash-Next Decode (64 Requests)",
        "fn_verify_c64": "Flash-Next MTP Verify (64 Requests)",
    }
    data = [
        {"window": name[r["conv_op"]], "arm": ARM[r["arm"]], "us": float(r["mean_us_per_call"])}
        for r in rows("sglang_conv_in_server_b300.csv")
    ]
    panels = []
    for window in name.values():
        sel = [d for d in data if d["window"] == window]
        mean = {a: sum((d["us"] for d in sel if d["arm"] == a)) / 2 for a in (STOCK, PLUGIN)}
        edge = {a: max((d["us"] for d in sel if d["arm"] == a)) for a in (STOCK, PLUGIN)}
        means = [{"arm": a, "us": v, "edge": edge[a], "t": f"{v:.1f} µs"} for a, v in mean.items()]
        panels.append(
            {
                "title": {
                    "text": window,
                    "fontSize": 14,
                    "anchor": "middle",
                    "offset": 8,
                },
                "width": 330,
                "height": 90,
                "layer": [
                    {
                        "data": {"values": means},
                        "mark": {"type": "bar", "height": 20, "opacity": 0.85},
                        "encoding": {
                            "x": {
                                "field": "us",
                                "type": "quantitative",
                                "axis": {"title": "Latency per Call (µs)", "tickCount": 5},
                            },
                            "y": {
                                "field": "arm",
                                "type": "nominal",
                                "sort": [STOCK, PLUGIN],
                                "axis": {"title": None, "ticks": False, "domain": False},
                            },
                            "color": {
                                "field": "arm",
                                "type": "nominal",
                                "scale": PAIR,
                                "legend": None,
                            },
                        },
                    },
                    {
                        "data": {"values": sel},
                        "mark": {"type": "point", "filled": True, "size": 40, "color": INK},
                        "encoding": {
                            "x": {"field": "us", "type": "quantitative"},
                            "y": {"field": "arm", "type": "nominal", "sort": [STOCK, PLUGIN]},
                        },
                    },
                    {
                        "data": {"values": means},
                        "mark": {
                            "type": "text",
                            "align": "left",
                            "dx": 8,
                            "font": FONT,
                            "fontSize": 13,
                            "color": INK,
                        },
                        "encoding": {
                            "x": {"field": "edge", "type": "quantitative"},
                            "y": {"field": "arm", "type": "nominal", "sort": [STOCK, PLUGIN]},
                            "text": {"field": "t"},
                        },
                    },
                ],
            }
        )
    spec = {
        "vconcat": [
            {"hconcat": panels[i : i + 2], "spacing": 50} for i in range(0, len(panels), 2)
        ],
        "spacing": 28,
    }
    save(spec, "e2e_conv_kernel_in_server")


WORKLOADS = (
    ("16K Prompt", "prefill_16k", "mean_latency_ms", 1),
    ("2K Prompt", "prefill_2k", "mean_latency_ms", 1),
    ("1 Request", "decode_c1", "output_tok_per_s", -1),
    ("16 Requests", "decode_c16", "output_tok_per_s", -1),
    ("64 Requests", "decode_c64", "output_tok_per_s", -1),
)


def lines(prefix, modes, skip=(), ttft=False):
    """Build the rows of a serving figure.

    Args:
        prefix (str): Workload prefix of the model in sglang_e2e_b300.csv.
        modes (tuple): Pairs of a mode label and its workload infix.
        skip (tuple): Pairs of mode label and workload that were not measured.
        ttft (bool): Whether the prompt rows use the median time to first token.

    Returns:
        list: Tuples of row label, workload, metric and the power that turns the metric into a
        time (1 for a latency, -1 for a throughput).
    """
    out = []
    for mode, infix in modes:
        for what, workload, metric, power in WORKLOADS:
            if (mode, workload) not in skip:
                if ttft and power == 1:
                    metric = "median_ttft_ms"
                out.append((f"{what}|({mode})", f"{prefix}{infix}{workload}", metric, power))
    return out


QWEN27 = lines(
    "",
    (("MTP", ""), ("No MTP", "nospec_")),
    skip=(("MTP", "decode_c64"), ("No MTP", "prefill_2k")),
)
FLASH_NEXT = lines(
    "fn_",
    (("MTP", "mtp_"), ("No MTP", "nospec_")),
    skip=(("MTP", "prefill_2k"), ("No MTP", "prefill_2k")),
    ttft=True,
)


def serving(rows_, name):
    """Render every serving session relative to the mean of the stock sessions.

    Args:
        rows_ (list): Tuples of row label, workload, metric and the power that turns the metric
            into a time (1 for a latency, -1 for a throughput).
        name (str): Output filename stem.
    """
    value = {}
    for r in rows("sglang_e2e_b300.csv"):
        value[r["arm"], r["session"], r["workload"], r["metric"]] = float(r["value"])
    data, notes = ([], [])
    for label, workload, metric, power in rows_:
        get = lambda arm, s: value[arm, str(s), workload, metric] ** power
        base = (get("stock", 1) + get("stock", 2)) / 2
        for arm in ("stock", "plugin"):
            for s in (1, 2):
                data.append({"row": label, "arm": ARM[arm], "pct": 100 * (get(arm, s) / base - 1)})
        change = 100 * ((get("plugin", 1) + get("plugin", 2)) / 2 / base - 1)
        notes.append({"row": label, "t": f"{change:+.1f}%"})
    order = [line[0] for line in rows_]
    span = max(2.0, 0.5 * (int(max(abs(d["pct"]) for d in data) / 0.5) + 1))
    spec = {
        "width": 560,
        "height": 44 * len(order),
        "layer": [
            {
                "data": {"values": [{"x": 0}]},
                "mark": {"type": "rule", "strokeDash": [5, 4], "color": INK, "strokeWidth": 1},
                "encoding": {"x": {"field": "x", "type": "quantitative"}},
            },
            {
                "data": {"values": data},
                "mark": {"type": "point", "filled": True, "size": 110, "opacity": 0.9},
                "encoding": {
                    "x": {
                        "field": "pct",
                        "type": "quantitative",
                        "scale": {"domain": [-span, span]},
                        "axis": {
                            "title": "Change in Latency (%)",
                            "tickCount": 9,
                        },
                    },
                    "y": {
                        "field": "row",
                        "type": "nominal",
                        "sort": order,
                        "axis": {
                            "title": None,
                            "ticks": False,
                            "domain": False,
                            "grid": True,
                            "labelPadding": 10,
                            "labelExpr": SPLIT,
                        },
                    },
                    "color": {
                        "field": "arm",
                        "type": "nominal",
                        "scale": PAIR,
                        "legend": {"title": None, "orient": "top", "direction": "horizontal"},
                    },
                    "shape": {
                        "field": "arm",
                        "type": "nominal",
                        "scale": {"domain": PAIR["domain"], "range": ["square", "circle"]},
                        "legend": {"title": None, "orient": "top", "direction": "horizontal"},
                    },
                },
            },
            {
                "data": {"values": notes},
                "mark": {
                    "type": "text",
                    "align": "right",
                    "font": FONT,
                    "fontSize": 13,
                    "fontWeight": "bold",
                    "color": INK,
                },
                "encoding": {
                    "x": {"value": 620},
                    "y": {"field": "row", "type": "nominal", "sort": order},
                    "text": {"field": "t"},
                },
            },
        ],
    }
    save(spec, name)


def accuracy():
    """Render the agreement with a float64 reference and the GSM8K accuracy of both arms."""
    label = {
        "conv_outputs_within_half_ulp_of_fp64": "Within 0.5 ULP|(vs. float64)",
        "conv_outputs_within_one_ulp_of_fp64": "Within 1 ULP|(vs. float64)",
        "gsm8k_1319_questions": "GSM8K|(5-shot)",
    }
    data = []
    for r in rows("sglang_accuracy_b300.csv"):
        if r["check"] in label:
            for arm in ("stock", "plugin"):
                v = float(r[arm])
                text = f"{v:.2f}%" if 99.9 < v < 100 else f"{v:.1f}%"
                data.append({"row": label[r["check"]], "arm": ARM[arm], "v": v, "t": text})
    order = list(label.values())
    spec = {
        "data": {"values": data},
        "width": 520,
        "height": {"step": 22},
        "facet": {
            "row": {
                "field": "row",
                "type": "nominal",
                "sort": order,
                "header": {
                    "title": None,
                    "labelAngle": 0,
                    "labelAlign": "left",
                    "labelOrient": "left",
                },
            }
        },
        "spec": {
            "layer": [
                {
                    "mark": {"type": "bar", "height": 16, "opacity": 0.85},
                    "encoding": {
                        "x": {
                            "field": "v",
                            "type": "quantitative",
                            "scale": {"domain": [0, 100]},
                            "axis": {"title": "Outputs / Accuracy (%)", "tickCount": 5},
                        },
                        "y": {
                            "field": "arm",
                            "type": "nominal",
                            "sort": [STOCK, PLUGIN],
                            "axis": None,
                        },
                        "color": {
                            "field": "arm",
                            "type": "nominal",
                            "scale": PAIR,
                            "legend": {
                                "title": None,
                                "orient": "bottom",
                                "direction": "horizontal",
                            },
                        },
                    },
                },
                {
                    "mark": {
                        "type": "text",
                        "align": "left",
                        "dx": 6,
                        "font": FONT,
                        "fontSize": 12,
                        "color": INK,
                    },
                    "encoding": {
                        "x": {"field": "v", "type": "quantitative"},
                        "y": {"field": "arm", "type": "nominal", "sort": [STOCK, PLUGIN]},
                        "text": {"field": "t"},
                    },
                },
            ]
        },
        "spacing": 14,
    }
    save(spec, "e2e_numerics")


def main():
    """Render every end-to-end figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    vlc.register_font_directory("/usr/share/texmf/fonts/opentype/public/lm")
    breakdown()
    conv_in_server()
    serving(QWEN27, "e2e_serving")
    serving(FLASH_NEXT, "e2e_serving_flash_next")
    accuracy()


if __name__ == "__main__":
    main()
