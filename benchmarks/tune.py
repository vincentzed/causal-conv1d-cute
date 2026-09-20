"""Benchmark kernel configurations and save the fastest to tuned.json.

Iterates through shape configurations for specified kernel kinds (defaulting to fwd and update),
optionally restricted to the models named with --models. Existing entries in tuned.json for the
selected cases are removed before profiling; all other entries are kept.

Candidate configurations are selected from fwd_candidates for forward kernels and update_candidates
for update kernels across standard and padded conv state layouts.

Each candidate is verified against a reference implementation using numerical checks. Update kernels
also require exact tensor equality on the updated conv state. Every passing configuration is timed
once, prefill on cold-cache latency and decode on the geometric mean of cold-cache and warm-cache
latency; the three fastest are timed again over three rounds and the best median is written to
tuned.json under the current GPU architecture, keyed by kernel kind, layout, channels, kernel
width, bias, and activation.
"""

import argparse
import json
from pathlib import Path
import torch
import harness as H
import shapes as S
from providers import Cute
from causal_conv1d_cute import api

OUT = Path(api.__file__).with_name("tuned.json")
ap = argparse.ArgumentParser()
ap.add_argument("kinds", nargs="*", default=["fwd", "update"])
ap.add_argument("--models", nargs="*", help="tune only these models and keep all other entries")
ap.add_argument("--variants", nargs="*", default=["cute", "cute_pad", "cute_ring"])
ARGS = ap.parse_args()
LAYOUT = {"cute": "std", "cute_pad": "pad", "cute_ring": "ring"}
KINDS = ARGS.kinds
CASES = S.cases(models=ARGS.models, kinds=KINDS)
full = json.loads(OUT.read_text()) if OUT.exists() else {}
table = full.setdefault(api.arch(), {})
stale = {
    api._key(c.kind, layout, c.D, c.W, c.bias, c.act)
    for c in CASES
    for layout in ([c.layout] if c.kind == "fwd" else [LAYOUT[v] for v in ARGS.variants])
}
for k in [k for k in table if k in stale]:
    del table[k]
log = []


def score(fn, kind, rounds):
    """Return the latency that ranks a configuration, in microseconds.

    Prefill is ranked on cold-cache latency. Decode is ranked on the geometric mean of cold-cache
    and warm-cache latency: a serving engine reads conv states that were last touched a full
    model step earlier, but the activations and weights it combines them with are often still
    cached, so neither measurement alone describes it.
    """
    cold = H.time_us(fn, rounds=rounds, dry=5, rep=20)
    if kind == "fwd":
        return cold
    return (cold * H.time_us(fn, rounds=rounds, cold=False, dry=5, rep=20)) ** 0.5


for case in CASES:
    t = H.make_inputs(case)
    ref = H.reference(case, t)
    for pname in ["cute"] if case.kind == "fwd" else ARGS.variants:
        p = Cute(pname)
        if case.kind == "fwd":
            cands, layout = (api.fwd_candidates(case.layout, case.W, case.B, case.L), case.layout)
        else:
            layout = LAYOUT[pname]
            cands = api.CausalConv1d.update_candidates(layout, case.D)
        best = None
        timed = []
        for cfg in cands:
            p.force = cfg
            try:
                made = p.fwd(case, t) if case.kind == "fwd" else p.update(case, t)
                made[0]()
                torch.cuda.synchronize()
                st = H.check(case, t, made[1](), ref if case.kind == "fwd" else ref[0])[0]
                if case.kind == "update" and (not torch.equal(made[2](), ref[1])):
                    st = "WRONG(state)"
                if st != "ok":
                    log.append(f"REJECT {case.label} {pname} {cfg}: {st}")
                    continue
                us = score(made[0], case.kind, 1)
            except Exception as e:
                log.append(
                    f"ERROR {case.label} {pname} {cfg}: {type(e).__name__}: {str(e).strip().splitlines()[0][:80]}"
                )
                continue
            timed.append((us, cfg, made[0]))
        for _, cfg, fn in sorted(timed, key=lambda r: r[0])[:3]:
            us = score(fn, case.kind, 3)
            if best is None or us < best[0]:
                best = (us, cfg)
        if best:
            key = api._key(case.kind, layout, case.D, case.W, case.bias, case.act)
            table.setdefault(key, []).append(
                {"B": case.B, "L": case.L, "cfg": list(best[1]), "us": round(best[0], 3)}
            )
            print(f"{case.label:<62} {pname:<9} -> {best[1]}  {best[0]:8.2f} us", flush=True)
OUT.write_text(json.dumps(full, indent=1))
print(f"\nwrote {OUT} ({sum((len(v) for v in table.values()))} entries)")
print("\n".join(log) if log else "no rejects/errors")
