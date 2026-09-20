import pytest
import torch
from causal_conv1d_cute import CausalConv1d, causal_conv1d_fn
from causal_conv1d_cute import fwd as F
from causal_conv1d_cute import testing as T

SHAPES = [(1, 16), (1, 17), (3, 33), (2, 1000), (1, 2047), (2, 512)]


def _run(case):
    t = T.make_inputs(case)
    out = causal_conv1d_fn(t["x"], t["w"], t["b"], case.act)
    torch.cuda.synchronize()
    return T.check(case, t, out, T.reference(case, t))


@pytest.mark.parametrize("layout", ["bdl", "btd"])
@pytest.mark.parametrize("act", [None, "silu"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("W", [2, 3, 4, 5, 6, 7, 8])
def test_all_widths(W, bias, act, layout):
    for B, L in SHAPES:
        status, _, bwd = _run(T.Case("t", "fwd", 64, W, B, L, bias, act, layout))
        assert status == "ok", (B, L, status, bwd)


@pytest.mark.parametrize("layout", ["bdl", "btd"])
@pytest.mark.parametrize(
    "D,W,bias,act", [(1536, 7, True, None), (2048, 3, False, None), (2048, 4, False, "silu")]
)
def test_real_shapes(D, W, bias, act, layout):
    for B, L in SHAPES:
        status, _, bwd = _run(T.Case("t", "fwd", D, W, B, L, bias, act, layout))
        assert status == "ok", (B, L, status, bwd)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("layout", ["bdl", "btd"])
def test_fp16(dtype, layout):
    status, _, bwd = _run(T.Case("t", "fwd", 1536, 7, 2, 512, True, "silu", layout, dtype))
    assert status == "ok", (status, bwd)


@pytest.mark.parametrize("S", [2, 3, 4, 5, 8])
def test_strip_shorter_than_lookback_stays_in_bounds(S):
    case = T.Case("t", "fwd", 1536, 7, 3, 64, True, None, "btd")
    t = T.make_inputs(case)
    wtm = F.pack_weight_timemajor(t["w"], t["b"])
    out = F.causal_conv1d_fwd_channellast_strip(
        t["x"], wtm, 7, True, None, strip=S, bs=64, hoist=True
    )
    torch.cuda.synchronize()
    assert T.check(case, t, out, T.reference(case, t))[0] == "ok"


def test_in_place_is_refused():
    case = T.Case("t", "fwd", 64, 4, 1, 64, False, None, "bdl")
    t = T.make_inputs(case)
    with pytest.raises(ValueError):
        CausalConv1d(t["w"], None, None).fwd(t["x"], out=t["x"])
