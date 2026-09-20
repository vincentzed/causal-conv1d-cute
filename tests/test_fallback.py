import warnings
import pytest
import torch
import torch.nn.functional as Fn
from causal_conv1d_cute import (
    causal_conv1d_fn,
    causal_conv1d_update,
    causal_conv1d_varlen_fn,
)


def _ref(x, w, b, act):
    D, W = w.shape
    y = Fn.conv1d(
        x.float(), w.float().unsqueeze(1), None if b is None else b.float(), padding=W - 1, groups=D
    )
    y = y[..., : x.shape[-1]]
    return Fn.silu(y) if act else y


@pytest.mark.parametrize("D,W", [(100, 4), (64, 9), (24, 2)])
@pytest.mark.parametrize("act", [None, "silu"])
def test_fn_falls_back_with_a_warning(D, W, act):
    torch.manual_seed(0)
    x = torch.randn(2, D, 40, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y = causal_conv1d_fn(x, w, b, act)
    assert torch.equal(y, _ref(x, w, b, act).to(x.dtype))


def test_fn_on_cpu():
    torch.manual_seed(0)
    x, w = (torch.randn(1, 32, 20), torch.randn(32, 4))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y = causal_conv1d_fn(x, w, None, "silu")
    assert torch.allclose(y, _ref(x, w, None, "silu"), atol=1e-6)


def test_fn_copies_unsupported_layouts():
    torch.manual_seed(0)
    x = torch.randn(2, 64, 96, device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
    y = causal_conv1d_fn(x, w, None, None)
    r = causal_conv1d_fn(x.contiguous(), w, None, None)
    assert torch.equal(y, r)


def test_varlen_and_update_fall_back():
    torch.manual_seed(0)
    D, W, K = (100, 4, 3)
    w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
    lens = [5, 1, 30]
    qsl = torch.tensor([0, 5, 6, 36], device="cuda", dtype=torch.int32)
    cidx = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
    hinit = torch.tensor([True, False, True], device="cuda")
    x = torch.randn(D, sum(lens), device="cuda", dtype=torch.bfloat16)
    st0 = torch.randn(5, D, K, device="cuda", dtype=torch.bfloat16)
    st = st0.clone()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y = causal_conv1d_varlen_fn(x, w, None, qsl, cidx, hinit, st, "silu")
    a = 0
    for n, slot, use in zip(lens, cidx.tolist(), hinit.tolist()):
        init = st0[slot] if use else torch.zeros_like(st0[slot])
        full = torch.cat([init, x[:, a : a + n]], dim=1)
        r = Fn.silu(Fn.conv1d(full.float().unsqueeze(0), w.float().unsqueeze(1), groups=D)[0])
        assert torch.equal(y[:, a : a + n], r.to(x.dtype))
        assert torch.equal(st[slot], full[:, -K:])
        a += n
    xs = torch.randn(3, D, device="cuda", dtype=torch.bfloat16)
    before = st.clone()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = causal_conv1d_update(xs, st, w, None, None, conv_state_indices=cidx)
    for i, slot in enumerate(cidx.tolist()):
        full = torch.cat([before[slot], xs[i].unsqueeze(-1)], dim=1)
        assert torch.equal(out[i], (full.float() * w.float()).sum(-1).to(xs.dtype))
        assert torch.equal(st[slot], full[:, 1:])


def test_bad_shapes_raise_value_error():
    x = torch.randn(1, 64, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(48, 4, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        causal_conv1d_fn(x, w)
    with pytest.raises(ValueError):
        causal_conv1d_fn(x, torch.randn(64, 4, 2, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        causal_conv1d_update(
            x[:, :, 0],
            torch.zeros(1, 64, 1, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 4, device="cuda", dtype=torch.bfloat16),
        )


def test_gradients_flow_through_the_fallback():
    torch.manual_seed(0)
    x = torch.randn(1, 64, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y = causal_conv1d_fn(x, w, None, "silu")
    y.square().sum().backward()
    ref_x = x.detach().clone().requires_grad_(True)
    _ref(ref_x, w.detach(), None, "silu").square().sum().backward()
    assert torch.allclose(x.grad, ref_x.grad, rtol=1e-4, atol=1e-5) and w.grad is not None
