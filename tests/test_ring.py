import random
import warnings
import pytest
import torch
import torch.nn.functional as Fn
from causal_conv1d_cute import causal_conv1d_update_ring, from_ring, to_ring

PAD = -1


def _step(ordered, x, w, b, act):
    full = torch.cat([ordered.float(), x.float().unsqueeze(-1)], dim=-1)
    y = (full * w.float()).sum(-1) + (0 if b is None else b.float())
    l1 = (full.abs() * w.float().abs()).sum(-1) + (0 if b is None else b.float().abs())
    return (Fn.silu(y) if act else y, l1, full[..., 1:].to(ordered.dtype))


def test_round_trip_between_layouts():
    torch.manual_seed(0)
    st = torch.randn(7, 32, 3, device="cuda")
    n = torch.randint(0, 50, (7,), device="cuda", dtype=torch.int32)
    ring = to_ring(st, n)
    assert ring.shape == (7, 3, 32) and torch.equal(from_ring(ring, n), st)


@pytest.mark.parametrize("seed", range(40))
def test_ring_decode_matches_ordered_state(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    D = random.choice([64, 1024, 2048, 10240])
    W = random.choice([2, 3, 4, 4, 4, 5, 8])
    B = random.choice([1, 2, 5, 48])
    dtype = random.choice([torch.bfloat16, torch.bfloat16, torch.float16, torch.float32])
    act = random.choice([None, "silu"])
    indexed = random.random() < 0.5
    K = W - 1
    slots = B + 4 if indexed else B
    w = torch.randn(D, W, device="cuda", dtype=dtype)
    b = torch.randn(D, device="cuda", dtype=dtype) if random.random() < 0.5 else None
    ordered = torch.randn(slots, D, K, device="cuda", dtype=dtype)
    seen = torch.randint(0, 97, (slots,), device="cuda", dtype=torch.int32)
    ring = to_ring(ordered, seen).contiguous()
    idx = None
    rows = torch.arange(B, device="cuda")
    if indexed:
        idx = torch.tensor(random.sample(range(slots), B), device="cuda", dtype=torch.int32)
        rows = idx.long()
    pads = [indexed and B > 1 and random.random() < 0.25 for _ in range(B)]
    if indexed and any(pads):
        idx[torch.tensor(pads, device="cuda")] = PAD
    live = torch.tensor([not p for p in pads], device="cuda")
    tol = 1.1 * (W + 2) * (2.0**-8 if dtype != torch.float32 else 2.0**-20)
    for _ in range(5):
        x = torch.randn(B, D, device="cuda", dtype=dtype)
        lens = seen[rows.clamp_min(0)] if indexed else seen
        y = causal_conv1d_update_ring(x, ring, w, b, act, lens, idx, PAD)
        torch.cuda.synchronize()
        ref, l1, new = _step(ordered[rows[live]], x[live], w, b, act)
        assert ((y[live].float() - ref).abs() / l1.clamp_min(1e-06)).max() <= tol
        ordered[rows[live]] = new
        seen[rows[live]] += 1
        assert torch.equal(from_ring(ring, seen), ordered)


def test_ring_falls_back_for_unsupported_shapes():
    torch.manual_seed(0)
    D, W = (100, 4)
    w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
    ordered = torch.randn(3, D, W - 1, device="cuda", dtype=torch.bfloat16)
    seen = torch.tensor([4, 5, 9], device="cuda", dtype=torch.int32)
    ring = to_ring(ordered, seen).contiguous()
    x = torch.randn(3, D, device="cuda", dtype=torch.bfloat16)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y = causal_conv1d_update_ring(x, ring, w, None, "silu", seen)
    ref, _, new = _step(ordered, x, w, None, "silu")
    assert torch.equal(y, ref.to(x.dtype)) and torch.equal(from_ring(ring, seen + 1), new)
