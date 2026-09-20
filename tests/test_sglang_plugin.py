import math
import random
import pytest
import torch
from causal_conv1d_cute import sglang_compat as C

sgl = pytest.importorskip("sglang.srt.layers.attention.mamba.causal_conv1d")
triton_ops = pytest.importorskip("sglang.kernels.ops.mamba.causal_conv1d_triton")

PAD = -1
bf = torch.bfloat16


def _rel(a, b):
    return float(((a.float() - b.float()).abs() / a.float().abs().clamp_min(1.0)).max())


@pytest.mark.parametrize("seed", range(40))
def test_prefill_matches_stock_on_padded_strided_input(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    D, W, cols = random.choice(
        [(10240, 4, 16384), (2048, 4, 2048), (1024, 3, 1040), (6144, 4, 8192), (2560, 4, 4120)]
    )
    nseq = random.choice([1, 1, 1, 2, 5, 16])
    lens = [random.choice([1, 2, 3, 7, 40, 129, 512, 1000]) for _ in range(nseq)]
    real = sum(lens)
    rows = max(real + random.choice([0, 0, 1, 13, 200]), 16)
    x = torch.randn(rows, cols, device="cuda", dtype=bf)[:, :D].t()
    w = torch.randn(D, W, device="cuda", dtype=bf) * 0.3
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    cidx = torch.tensor(random.sample(range(1, 40), nseq), device="cuda", dtype=torch.int32)
    if nseq > 1 and random.random() < 0.3:
        cidx[random.randrange(nseq)] = PAD
    hinit = torch.tensor([random.random() < 0.6 for _ in range(nseq)], device="cuda")
    st0 = torch.randn(40, D, W - 1, device="cuda", dtype=bf)
    sa, sb = (st0.clone(), st0.clone())
    kw = dict(
        query_start_loc=qsl,
        cache_indices=cidx,
        has_initial_state=hinit,
        activation="silu",
        seq_lens_cpu=list(lens),
    )
    before = dict(C.stats)
    ya = sgl.causal_conv1d_fn(x, w, None, conv_states=sa, **kw)
    yb = C.around_fn(sgl.causal_conv1d_fn, x, w, None, conv_states=sb, **kw)
    torch.cuda.synchronize()
    assert C.stats["fn_fast"] == before["fn_fast"] + 1
    assert torch.equal(sa, sb)
    a = 0
    for n, slot in zip(lens, cidx.tolist()):
        if slot != PAD:
            assert _rel(ya[:, a : a + n], yb[:, a : a + n]) <= 0.04
        a += n


@pytest.mark.parametrize("seed", range(24))
def test_verify_and_decode_match_stock(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    D, W = random.choice([(10240, 4), (2048, 4), (1024, 3)])
    K = W - 1
    B = random.choice([1, 3, 16, 48])
    L = random.choice([1, 4, 4, 6])
    w = torch.randn(D, W, device="cuda", dtype=bf) * 0.3
    st0 = torch.randn(64, D, K, device="cuda", dtype=bf)
    idx = torch.tensor(random.sample(range(64), B), device="cuda", dtype=torch.int32)
    if B > 1:
        idx[random.randrange(B)] = PAD
    sa, sb = (st0.clone(), st0.clone())
    kw = dict(conv_state_indices=idx)
    pa = pb = None
    if L > 1:
        x = torch.randn(B, L, D, device="cuda", dtype=bf).transpose(1, 2)
        PW = L + K - 1
        pa, pb = (torch.zeros(B + 1, D, PW, device="cuda", dtype=bf) for _ in range(2))
        view = lambda p: p.as_strided((B + 1, L, D, K), (D * PW, 1, PW, 1))
        iidx = torch.arange(B, device="cuda", dtype=torch.int32)
        ka = dict(kw, intermediate_conv_window=view(pa), intermediate_state_indices=iidx)
        kb = dict(kw, intermediate_conv_window=view(pb), intermediate_state_indices=iidx)
    else:
        x = torch.randn(B, D, device="cuda", dtype=bf)
        ka = kb = kw
    up = triton_ops.causal_conv1d_update
    before = dict(C.stats)
    ya = up(x, sa, w, None, "silu", **ka)
    yb = C.around_update(up, x, sb, w, None, "silu", **kb)
    torch.cuda.synchronize()
    assert C.stats["update_fast"] == before["update_fast"] + 1
    assert yb.shape == ya.shape and yb.stride() == ya.stride()
    assert torch.equal(sa, sb)
    if pa is not None:
        assert torch.equal(pa, pb)
    live = idx != PAD
    assert _rel(ya[live], yb[live]) <= 0.04


@pytest.mark.parametrize("seed", range(24))
def test_fused_decode_unpack_matches_stock(seed):
    fused = pytest.importorskip("sglang.kernels.ops.attention.triton_gdn_fused_proj")
    stock = fused.fused_qkvzba_causal_conv1d_update_contiguous
    stock = getattr(stock, "__wrapped__", stock)
    random.seed(seed)
    torch.manual_seed(seed)
    heads, head_dim = random.choice([(48, 128), (12, 128), (16, 128), (6, 32)])
    v_dim = heads * head_dim
    qkv_dim = v_dim + random.choice([1024, 4096]) if v_dim >= 1024 else 2048
    W = random.choice([3, 4, 4])
    B = random.choice([1, 2, 7, 33, 64])
    act = random.choice([None, "silu"])
    dtype = torch.bfloat16
    w = torch.randn(qkv_dim, W, device="cuda", dtype=dtype) * 0.3
    bias = torch.randn(qkv_dim, device="cuda", dtype=dtype) if random.random() < 0.4 else None
    state = torch.randn(B + 5, qkv_dim, W - 1, device="cuda", dtype=dtype)
    if random.random() < 0.5:
        qkvz = torch.randn(B, qkv_dim + v_dim, device="cuda", dtype=dtype)
        ba = torch.randn(B, 2 * heads, device="cuda", dtype=dtype)
    else:
        fused = torch.randn(B, qkv_dim + v_dim + 2 * heads, device="cuda", dtype=dtype)
        qkvz, ba = (fused[:, : qkv_dim + v_dim], fused[:, qkv_dim + v_dim :])
    idx = torch.tensor(random.sample(range(B + 5), B), device="cuda", dtype=torch.int32)
    if B > 2:
        idx[random.randrange(B)] = C.PAD_SLOT_ID
    if random.random() < 0.3:
        idx = idx.long()
    kw = dict(qkv_dim=qkv_dim, v_dim=v_dim, num_v_heads=heads, head_v_dim=head_dim, activation=act)
    s1, s2 = (state.clone(), state.clone())
    ref = stock(qkvz, ba, s1, w, bias, idx, **kw)
    before = dict(C.stats)
    got = C.around_unpack(stock, qkvz, ba, s2, w, bias, idx, **kw)
    torch.cuda.synchronize()
    served = C.unpack.unpack_reason(qkvz, ba, s2, w, bias, heads) is None
    assert served == (math.gcd(16, qkvz.stride(0)) >= 4)
    assert C.stats["unpack_fast"] == before["unpack_fast"] + int(served)
    assert [t.shape for t in got] == [t.shape for t in ref]
    assert torch.equal(s1, s2)
    for g, r in zip(got[1:], ref[1:]):
        assert torch.equal(g, r)
    scale = ref[0].float().abs().clamp_min(1.0)
    assert ((got[0].float() - ref[0].float()).abs() / scale).max() <= 0.02
