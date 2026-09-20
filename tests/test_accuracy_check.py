import pytest
import torch
import torch.nn.functional as F
from causal_conv1d_cute import testing as T


@pytest.mark.parametrize(
    "D,W,act,bias", [(1536, 7, None, True), (2048, 3, None, False), (1024, 4, "silu", True)]
)
def test_check_classifies_known_outputs(D, W, act, bias):
    case = T.Case("check", "fwd", D, W, 2, 512, bias, act, "bdl")
    t = T.make_inputs(case)
    ref = T.reference(case, t)
    x, w, b = (t["x"], t["w"], t["b"])
    fin = (lambda y: F.silu(y)) if act else lambda y: y
    xp = F.pad(x, (W - 1, 0))
    acc = b.view(1, D, 1).expand(2, D, 512).clone() if bias else torch.zeros_like(x)
    for k in range(W):
        acc = acc + w[:, k].view(1, D, 1) * xp[..., k : k + 512]
    w_drop = w.clone()
    w_drop[:, 0] = 0
    dropped = fin(
        F.conv1d(
            x.float(),
            w_drop.float().unsqueeze(1),
            b.float() if bias else None,
            padding=W - 1,
            groups=D,
        )[..., :512]
    ).to(case.dtype)
    assert T.check(case, t, ref.to(case.dtype), ref)[0] == "ok"
    assert T.check(case, t, fin(acc), ref)[0] == "lowprec"
    assert T.check(case, t, dropped, ref)[0] == "WRONG"
    assert T.check(case, t, torch.roll(ref.to(case.dtype), 1, dims=-1), ref)[0] == "WRONG"
