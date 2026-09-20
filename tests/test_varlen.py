import random
import pytest
import torch
import torch.nn.functional as Fn
from causal_conv1d_cute import causal_conv1d_varlen_fn

PAD = -1


def _ref(xs, w, b, init, act):
    D, W = w.shape
    full = torch.cat([init if init is not None else xs.new_zeros(D, W - 1), xs], dim=1)
    y = Fn.conv1d(full.unsqueeze(0), w.unsqueeze(1), b, groups=D)[0]
    return Fn.silu(y) if act else y


def _batch(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    D, W = (random.choice([64, 1024, 1536]), random.choice([2, 3, 4, 7, 8]))
    K, act, has_bias = (W - 1, random.choice([None, "silu"]), random.random() < 0.5)
    nseq = random.choice([1, 2, 5, 9])
    lens = [random.choice([1, 2, 3, K, K + 1, 17, 64, 333, 1000]) for _ in range(nseq)]
    if sum(lens) < 16:
        lens[0] += 32
    slots = nseq + 3
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    cidx = torch.tensor(random.sample(range(slots), nseq), device="cuda", dtype=torch.int32)
    pads = [random.random() < 0.15 for _ in range(nseq)]
    if all(pads):
        pads[0] = False
    cidx[torch.tensor(pads, device="cuda")] = PAD
    hinit = torch.tensor([random.random() < 0.5 for _ in range(nseq)], device="cuda")
    x = torch.randn(D, sum(lens), device="cuda", dtype=torch.bfloat16)
    w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(D, device="cuda", dtype=torch.bfloat16) if has_bias else None
    st0 = torch.randn(slots, D, K, device="cuda", dtype=torch.bfloat16)
    return (D, W, K, act, nseq, qsl, cidx, pads, hinit, x, w, b, st0)


@pytest.mark.parametrize("seed", range(24))
def test_varlen_outputs_and_states(seed):
    D, W, K, act, nseq, qsl, cidx, pads, hinit, x, w, b, st0 = _batch(seed)
    st = st0.clone()
    y = causal_conv1d_varlen_fn(x, w, b, qsl, cidx, hinit, st, act, PAD)
    torch.cuda.synchronize()
    exp = st0.clone()
    for i in range(nseq):
        if pads[i]:
            continue
        a, e, sl = (int(qsl[i]), int(qsl[i + 1]), int(cidx[i]))
        init = st0[sl] if bool(hinit[i]) else None
        r = _ref(
            x[:, a:e].float(),
            w.float(),
            None if b is None else b.float(),
            None if init is None else init.float(),
            act,
        )
        l1 = _ref(
            x[:, a:e].float().abs(),
            w.float().abs(),
            None if b is None else b.float().abs(),
            None if init is None else init.float().abs(),
            None,
        )
        assert ((y[:, a:e].float() - r).abs() / l1.clamp_min(1e-06)).max() <= 1.1 * (
            W + 2
        ) * 2.0 ** (-8)
        old = st0[sl] if bool(hinit[i]) else torch.zeros_like(st0[sl])
        exp[sl] = torch.cat([old, x[:, a:e]], dim=1)[:, -K:]
    assert torch.equal(st, exp)


@pytest.mark.parametrize("seed", range(12))
def test_states_bit_exact_against_sglang(seed):
    sgl = pytest.importorskip("sglang.srt.layers.attention.mamba.causal_conv1d")
    D, W, K, act, nseq, qsl, cidx, pads, hinit, x, w, b, st0 = _batch(seed)
    if W > 4:
        pytest.skip("sglang's kernel supports widths 2-4 only")
    st_a, st_b = (st0.clone(), st0.clone())
    sgl.causal_conv1d_fn(
        x.clone(),
        w,
        b,
        query_start_loc=qsl,
        cache_indices=cidx,
        has_initial_state=hinit,
        conv_states=st_a,
        activation=act,
        pad_slot_id=PAD,
    )
    causal_conv1d_varlen_fn(x, w, b, qsl, cidx, hinit, st_b, act, PAD)
    torch.cuda.synchronize()
    assert torch.equal(st_a, st_b)
