import sys
import types
import pytest
import torch
from causal_conv1d_cute import _patch


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def test_wrap_rebinds_every_importer_and_is_idempotent():
    def original(x, scale=1):
        return x * scale

    lib = _module("_cc1d_lib", fn=original)
    user_a = _module("_cc1d_user_a", fn=original)
    user_b = _module("_cc1d_user_b", renamed=original, other=len)
    calls = []

    def around(orig, x, scale=1):
        calls.append(x)
        return orig(x, scale) + 1

    try:
        assert _patch.wrap("_cc1d_lib.fn", around) == 3
        assert lib.fn(2, scale=3) == 7 and user_a.fn(1) == 2 and user_b.renamed(5) == 6
        assert user_b.other is len and calls == [2, 1, 5]
        assert lib.fn.__name__ == "original"
        assert _patch.wrap("_cc1d_lib.fn", around) == 0
        assert lib.fn(2) == 3
    finally:
        for name in ("_cc1d_lib", "_cc1d_user_a", "_cc1d_user_b"):
            sys.modules.pop(name, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_dao_compat_serves_inference_and_passes_on_the_rest():
    upstream = pytest.importorskip("causal_conv1d")
    from causal_conv1d_cute import dao_compat

    iface = sys.modules["causal_conv1d.causal_conv1d_interface"]
    torch.manual_seed(0)
    w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 256, 64, device="cuda", dtype=torch.bfloat16)
    ref = upstream.causal_conv1d_fn(x, w, activation="silu")
    dao_compat.install()
    before = dict(dao_compat.stats)
    y = iface.causal_conv1d_fn(x, w, activation="silu")
    assert dao_compat.stats["fn_fast"] == before["fn_fast"] + 1
    assert (y.float() - ref.float()).abs().max() <= 0.05 * ref.float().abs().max()
    xg = x.clone().requires_grad_(True)
    iface.causal_conv1d_fn(xg, w, activation="silu").sum().backward()
    assert dao_compat.stats["fn_fallback"] == before["fn_fallback"] + 1 and xg.grad is not None
