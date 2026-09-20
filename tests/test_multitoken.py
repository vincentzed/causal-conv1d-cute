import random
import pytest
import torch
import torch.nn.functional as Fn
from causal_conv1d_cute import causal_conv1d_update, causal_conv1d_varlen_fn

PAD = -1


def _conv(full, w, b, act):
    y = Fn.conv1d(full.unsqueeze(0), w.unsqueeze(1), b, groups=w.shape[0])[0]
    return Fn.silu(y) if act else y


def _inter(layout, slots, L, D, K, dtype):
    if layout is None:
        return (None, None)
    if layout == "dense":
        t = torch.full((slots, L, D, K), 7.0, device="cuda", dtype=dtype)
        return (t, t)
    PW = L + K - 1
    phys = torch.full((slots, D, PW), 7.0, device="cuda", dtype=dtype)
    view = phys.as_strided((slots, L, D, K), (D * PW, 1, PW, 1))
    return (phys, view)


def _case(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    D = random.choice([64, 256, 1024, 2048, 10240])
    W = random.choice([2, 3, 4, 4, 4])
    L = random.choice([1, 2, 4, 4, 5, 8])
    B = random.choice([1, 2, 7, 48])
    layout = random.choice([None, "dense", "dedup", "dedup"])
    dtype = random.choice([torch.bfloat16, torch.bfloat16, torch.float16, torch.float32])
    act = random.choice([None, "silu"])
    has_bias = random.random() < 0.5
    return (D, W, L, B, layout, dtype, act, has_bias)


@pytest.mark.parametrize("seed", range(64))
def test_multitoken_update(seed):
    D, W, L, B, layout, dtype, act, has_bias = _case(seed)
    K = W - 1
    slots = B + 5
    x = torch.randn(B, L, D, device="cuda", dtype=dtype).transpose(1, 2)
    w = torch.randn(D, W, device="cuda", dtype=dtype)
    b = torch.randn(D, device="cuda", dtype=dtype) if has_bias else None
    st0 = torch.randn(slots, D, K, device="cuda", dtype=dtype)
    idx = torch.tensor(random.sample(range(slots), B), device="cuda", dtype=torch.int32)
    iidx = torch.tensor(random.sample(range(slots), B), device="cuda", dtype=torch.int32)
    pads = [B > 1 and random.random() < 0.2 for _ in range(B)]
    if all(pads):
        pads[0] = False
    idx[torch.tensor(pads, device="cuda")] = PAD
    phys, inter = _inter(layout, slots, L, D, K, dtype)
    st = st0.clone()
    y = causal_conv1d_update(
        x,
        st,
        w,
        b,
        act,
        conv_state_indices=idx,
        pad_slot_id=PAD,
        intermediate_conv_window=inter,
        intermediate_state_indices=None if inter is None else iidx,
    )
    torch.cuda.synchronize()
    assert y.shape == x.shape and y.stride() == x.stride()
    exp = st0.clone()
    tol = 1.1 * (W + 2) * (2.0**-8 if dtype != torch.float32 else 2.0**-20)
    for i in range(B):
        if pads[i]:
            continue
        sl = int(idx[i])
        hist = torch.cat([st0[sl], x[i]], dim=1)
        r = _conv(hist.float(), w.float(), None if b is None else b.float(), act)
        l1 = _conv(
            hist.float().abs(), w.float().abs(), None if b is None else b.float().abs(), None
        )
        assert ((y[i].float() - r).abs() / l1.clamp_min(1e-06)).max() <= tol
        exp[sl] = hist[:, L:]
        if inter is not None:
            for t in range(L):
                assert torch.equal(inter[int(iidx[i]), t], hist[:, t + 1 : t + 1 + K])
    assert torch.equal(st, exp)
    if inter is not None:
        live = {int(iidx[i]) for i in range(B) if not pads[i]}
        for s in range(slots):
            if s not in live:
                assert bool((phys[s] == 7.0).all())


@pytest.mark.parametrize("seed", range(12))
def test_single_token_rows_of_wider_buffer(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    D, W, B = (random.choice([256, 2048, 10240]), random.choice([3, 4]), random.choice([1, 5, 48]))
    K = W - 1
    wide = torch.randn(B, D + 6144, device="cuda", dtype=torch.bfloat16)
    x = wide[:, :D]
    w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
    st0 = torch.randn(B + 2, D, K, device="cuda", dtype=torch.bfloat16)
    idx = torch.tensor(random.sample(range(B + 2), B), device="cuda", dtype=torch.int32)
    st = st0.clone()
    y = causal_conv1d_update(x, st, w, None, "silu", conv_state_indices=idx)
    ref_st = st0.clone()
    r = causal_conv1d_update(x.contiguous(), ref_st, w, None, "silu", conv_state_indices=idx)
    torch.cuda.synchronize()
    assert torch.equal(y, r) and torch.equal(st, ref_st)


@pytest.mark.parametrize("seed", range(16))
def test_varlen_channel_last_views(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    D, W = (random.choice([64, 1024, 10240]), random.choice([2, 3, 4]))
    K = W - 1
    cols = D + random.choice([0, 0, 16, 6144])
    nseq = random.choice([1, 3, 8])
    lens = [random.choice([1, 2, K, 17, 64, 333, 2048]) for _ in range(nseq)]
    if sum(lens) < 16:
        lens[0] += 32
    T = sum(lens)
    act, has_bias = (random.choice([None, "silu"]), random.random() < 0.5)
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    cidx = torch.tensor(random.sample(range(nseq + 2), nseq), device="cuda", dtype=torch.int32)
    hinit = torch.tensor([random.random() < 0.5 for _ in range(nseq)], device="cuda")
    wide = torch.randn(T, cols, device="cuda", dtype=torch.bfloat16)
    x = wide[:, :D].t()
    w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(D, device="cuda", dtype=torch.bfloat16) if has_bias else None
    st0 = torch.randn(nseq + 2, D, K, device="cuda", dtype=torch.bfloat16)
    st_a, st_b = (st0.clone(), st0.clone())
    keep = wide.clone()
    y = causal_conv1d_varlen_fn(x, w, b, qsl, cidx, hinit, st_a, act, PAD)
    r = causal_conv1d_varlen_fn(x.contiguous(), w, b, qsl, cidx, hinit, st_b, act, PAD)
    torch.cuda.synchronize()
    assert y.t().is_contiguous()
    assert torch.equal(wide, keep)
    assert torch.equal(st_a, st_b)
    l1 = x.float().abs().amax().item() * w.float().abs().sum(1).amax().item() + 1.0
    assert ((y.float() - r.float()).abs().max().item() / l1) <= 2.0**-6


@pytest.mark.parametrize("T", [16, 17, 100, 511, 512, 1023, 2048, 2049, 5000])
@pytest.mark.parametrize(
    "cols_extra,use_init,padded",
    [(0, True, False), (6144, True, False), (6144, False, False), (16, True, True)],
)
def test_single_sequence_single_launch(T, cols_extra, use_init, padded):
    torch.manual_seed(T)
    D, W, K = (10240 if cols_extra == 6144 else 1024, 4, 3)
    wide = torch.randn(T, D + cols_extra, device="cuda", dtype=torch.bfloat16)
    x = wide[:, :D].t()
    w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
    st0 = torch.randn(4, D, K, device="cuda", dtype=torch.bfloat16)
    st = st0.clone()
    qsl = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    cidx = torch.tensor([PAD if padded else 2], device="cuda", dtype=torch.int32)
    hinit = torch.tensor([use_init], device="cuda")
    y = causal_conv1d_varlen_fn(x, w, None, qsl, cidx, hinit, st, "silu", PAD)
    torch.cuda.synchronize()
    assert y.t().is_contiguous()
    if padded:
        assert torch.equal(st, st0)
        return
    init = st0[2] if use_init else torch.zeros_like(st0[2])
    full = torch.cat([init, x], dim=1)
    r = _conv(full.float(), w.float(), None, "silu")
    l1 = _conv(full.float().abs(), w.float().abs(), None, None)
    assert ((y.float() - r).abs() / l1.clamp_min(1e-06)).max() <= 1.1 * (W + 2) * 2.0**-8
    exp = st0.clone()
    exp[2] = full[:, -K:]
    assert torch.equal(st, exp)


@pytest.mark.parametrize(
    "rows,bos,eos",
    [
        (2048, 0, 2048),
        (2048, 0, 1777),
        (512, 0, 1),
        (512, 0, 2),
        (480, 0, 257),
        (1024, 0, 900),
        (64, 0, 3),
    ],
)
@pytest.mark.parametrize("use_init", [True, False])
def test_single_sequence_with_padding_rows(rows, bos, eos, use_init):
    torch.manual_seed(rows + eos)
    D, W, K = (10240, 4, 3)
    wide = torch.randn(rows, 16384, device="cuda", dtype=torch.bfloat16)
    x = wide[:, :D].t()
    w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
    st0 = torch.randn(4, D, K, device="cuda", dtype=torch.bfloat16)
    st = st0.clone()
    qsl = torch.tensor([bos, eos], device="cuda", dtype=torch.int32)
    cidx = torch.tensor([1], device="cuda", dtype=torch.int32)
    hinit = torch.tensor([use_init], device="cuda")
    y = causal_conv1d_varlen_fn(x, w, None, qsl, cidx, hinit, st, "silu", PAD)
    torch.cuda.synchronize()
    init = st0[1] if use_init else torch.zeros_like(st0[1])
    full = torch.cat([init, x[:, bos:eos]], dim=1)
    r = _conv(full.float(), w.float(), None, "silu")
    l1 = _conv(full.float().abs(), w.float().abs(), None, None)
    assert ((y[:, bos:eos].float() - r).abs() / l1.clamp_min(1e-06)).max() <= 1.1 * (
        W + 2
    ) * 2.0**-8
    exp = st0.clone()
    exp[1] = full[:, -K:]
    assert torch.equal(st, exp)
