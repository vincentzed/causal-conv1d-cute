import pytest
import torch
from causal_conv1d_cute import CausalConv1d, causal_conv1d_update
from causal_conv1d_cute import testing as T
from causal_conv1d_cute import update as U


@pytest.mark.parametrize("B", [1, 3, 64])
@pytest.mark.parametrize("act", [None, "silu"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("W", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("D", [64, 1536])
def test_standard_state(D, W, bias, act, B):
    case = T.Case("t", "update", D, W, B, 1, bias, act)
    t = T.make_inputs(case)
    ref_out, ref_state = T.reference(case, t)
    state = t["state"].clone()
    out = causal_conv1d_update(t["x"], state, t["w"], t["b"], act)
    torch.cuda.synchronize()
    assert T.check(case, t, out, ref_out)[0] == "ok"
    assert torch.equal(state, ref_state)


@pytest.mark.parametrize("B", [1, 8, 256])
@pytest.mark.parametrize(
    "D,W,bias,act", [(1536, 7, True, None), (2048, 3, False, None), (2048, 4, False, "silu")]
)
def test_padded_state(D, W, bias, act, B):
    case = T.Case("t", "update", D, W, B, 1, bias, act)
    t = T.make_inputs(case)
    ref_out, ref_state = T.reference(case, t)
    layer = CausalConv1d(t["w"], t["b"], act)
    P = layer.wpad.shape[1]
    hist = torch.randn(B, D, P - (W - 1), device="cuda", dtype=case.dtype)
    state = torch.cat([hist, t["state"]], dim=-1).contiguous()
    out = layer.update(t["x"], state)
    torch.cuda.synchronize()
    assert T.check(case, t, out, ref_out)[0] == "ok"
    assert torch.equal(state[..., P - (W - 1) :], ref_state)


def test_indices_and_padding():
    D, W, B, slots = (1536, 4, 8, 13)
    x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
    idx = torch.randperm(slots, device="cuda")[:B].to(torch.int32)
    idx[2] = -1
    st0 = torch.randn(slots, D, W - 1, device="cuda", dtype=torch.bfloat16)
    st = st0.clone()
    out = causal_conv1d_update(x, st, w, None, "silu", conv_state_indices=idx)
    torch.cuda.synchronize()
    exp = st0.clone()
    for b in range(B):
        if int(idx[b]) >= 0:
            exp[int(idx[b])] = torch.cat([st0[int(idx[b])][:, 1:], x[b].unsqueeze(-1)], dim=1)
    assert torch.equal(st, exp)
    live = idx >= 0
    full = torch.cat([st0[idx.clamp_min(0).long()].float(), x.float().unsqueeze(-1)], -1)
    ref = torch.nn.functional.silu((full * w.float()).sum(-1))
    assert (out[live].float() - ref[live]).abs().max() < 0.07


def test_in_place_kernel_refuses_inexact_launch_and_is_deterministic():
    D, W, B = (1536, 7, 1)
    x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
    wp = U.pad_weight(torch.randn(D, W, device="cuda", dtype=torch.bfloat16))
    st = torch.randn(B, D, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        U.causal_conv1d_update_padded_vec(x, st.clone(), wp, W, cv=16, bs=64)
    ref = torch.cat([st[..., 1:], x.unsqueeze(-1)], -1)
    for _ in range(100):
        s2 = st.clone()
        U.causal_conv1d_update_padded_vec(x, s2, wp, W, cv=16, bs=32)
        assert torch.equal(s2, ref)
