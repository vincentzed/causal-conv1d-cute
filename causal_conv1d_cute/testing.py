"""Provide test cases, reference outputs, and accuracy checks.

Common test configurations, input generators, float32 reference implementations, and numerical error
checks for causal 1D convolution kernels.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F

MANT_BITS = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}


@dataclass(frozen=True)
class Case:
    """Hold configuration parameters for causal 1D convolution kernels.

    Attributes:
        model: Architecture name.
        kind: Operation mode, 'fwd' for convolution or 'update' for single-step update.
        D: Number of channels.
        W: Kernel width.
        B: Batch size.
        L: Sequence length for forward execution.
        bias: Whether to add bias.
        act: Activation function name or None.
        layout: Memory layout for input tensor ('bdl' or 'btd').
        dtype: Data type of input and state tensors.
    """

    model: str
    kind: str
    D: int
    W: int
    B: int
    L: int = 1
    bias: bool = True
    act: str | None = None
    layout: str = "bdl"
    dtype: torch.dtype = torch.bfloat16

    @property
    def label(self):
        """Return a formatted string describing the configuration.

        Returns:
            str: String containing model, mode, dimensions, layout, and activation.
        """
        a = "silu" if self.act else "none"
        if self.kind == "fwd":
            return f"{self.model} fwd D={self.D} W={self.W} B={self.B} L={self.L} {self.layout} {a}"
        return f"{self.model} upd D={self.D} W={self.W} B={self.B} {a}"

    @property
    def bytes_moved(self):
        """Calculate minimum memory traffic in bytes.

        Assumes one read per input element and one write per output element.

        Returns:
            int: Total bytes transferred between global memory and registers.
        """
        n = self.B * self.D * self.L
        e = torch.empty((), dtype=self.dtype).element_size()
        if self.kind == "fwd":
            return 2 * n * e
        return (2 * self.B * self.D * (self.W - 1) + 2 * self.B * self.D) * e


def make_inputs(case: Case, seed=0):
    """Generate random CUDA input tensors for a test case.

    Args:
        case: Test case configuration.
        seed: Random seed for CUDA generator.

    Returns:
        dict: Dictionary containing generated tensors: 'x': (batch, channels, seqlen) for forward
            mode, or (batch, channels) for update mode. 'w': Filter weights of shape (channels,
            width). 'b': Bias of shape (channels,) if bias is enabled, else None. 'state': Conv
            state of shape (batch, channels, width - 1) for update mode.

    Raises:
        ValueError: If case.layout is not recognized.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = case.dtype
    w = torch.randn(case.D, case.W, device="cuda", dtype=dt, generator=g)
    b = torch.randn(case.D, device="cuda", dtype=dt, generator=g) if case.bias else None
    if case.kind == "fwd":
        if case.layout == "bdl":
            x = torch.randn(case.B, case.D, case.L, device="cuda", dtype=dt, generator=g)
        elif case.layout == "btd":
            x = torch.randn(case.B, case.L, case.D, device="cuda", dtype=dt, generator=g).transpose(
                1, 2
            )
        else:
            raise ValueError(case.layout)
        return dict(x=x, w=w, b=b)
    x = torch.randn(case.B, case.D, device="cuda", dtype=dt, generator=g)
    state = torch.randn(case.B, case.D, case.W - 1, device="cuda", dtype=dt, generator=g)
    return dict(x=x, w=w, b=b, state=state)


def reference(case: Case, t):
    """Compute reference output using float32 arithmetic.

    Args:
        case: Test case configuration.
        t: Dictionary containing input tensors 'w', 'b', 'x', and optional 'state'.

    Returns:
        For 'fwd' mode, float32 output tensor of shape (batch, channels, seqlen).
        For 'update' mode, tuple of: out: (batch, channels), float32 output tensor. next_state:
            (batch, channels, width - 1), updated conv state in case.dtype.
    """
    w32 = t["w"].float()
    b32 = t["b"].float() if t["b"] is not None else None
    if case.kind == "fwd":
        y = F.conv1d(t["x"].float(), w32.unsqueeze(1), b32, padding=case.W - 1, groups=case.D)[
            ..., : case.L
        ]
        return F.silu(y) if case.act else y
    full = torch.cat([t["state"].float(), t["x"].float().unsqueeze(-1)], dim=-1)
    y = (full * w32).sum(-1)
    if b32 is not None:
        y = y + b32
    return (F.silu(y) if case.act else y, full[..., 1:].to(case.dtype))


REF_MIN = 0.25


def l1_magnitude(case: Case, t):
    """Compute elementwise L1 scale in float32 for error analysis.

    Args:
        case: Test case configuration.
        t: Dictionary containing input tensors 'w', 'b', 'x', and optional 'state'.

    Returns:
        torch.Tensor: Float32 tensor of shape (batch, channels, seqlen) for forward mode or (batch,
            channels) for update mode, containing sum_k |w_k * x_k| + |b|.
    """
    w32 = t["w"].float().abs()
    b32 = t["b"].float().abs() if t["b"] is not None else None
    if case.kind == "fwd":
        return F.conv1d(
            t["x"].float().abs(), w32.unsqueeze(1), b32, padding=case.W - 1, groups=case.D
        )[..., : case.L]
    full = torch.cat([t["state"].float(), t["x"].float().unsqueeze(-1)], dim=-1).abs()
    m = (full * w32).sum(-1)
    return m + b32 if b32 is not None else m


def check(case: Case, t, got: torch.Tensor, ref32: torch.Tensor):
    """Validate kernel output against a float32 reference.

    Computes maximum error in units in the last place (ulp) for reference values with magnitude at
    least 0.25 to avoid cancellation artifacts.

    Computes backward error as maximum |got - ref| divided by (sum_k |w_k * x_k| + |b|) across all
    output elements. If backward error exceeds the analytical accumulation bound for the given
    kernel width and dtype precision, the status is 'WRONG'. Otherwise, returns 'ok' if ulp error is
    at most 1.5, or 'lowprec' if accumulated in reduced precision.

    Args:
        case: Test case configuration.
        t: Dictionary containing original input tensors.
        got: Kernel output tensor to evaluate.
        ref32: Unrounded float32 reference tensor.

    Returns:
        tuple: (status, ulp_err, bwd_err), where status is 'ok', 'lowprec', or 'WRONG'.
    """
    got, ref32 = (got.float(), ref32.float())
    err = (got - ref32).abs()
    keep = ref32.abs() >= REF_MIN
    exp = torch.floor(torch.log2(ref32.abs().clamp_min(REF_MIN)))
    ulp_err = (err / torch.pow(2.0, exp - MANT_BITS[case.dtype]))[keep].max().item()
    bwd_err = (err / l1_magnitude(case, t).clamp_min(1e-06)).max().item()
    half_ulp_rel = 2.0 ** (-(MANT_BITS[case.dtype] + 1))
    lowprec_bound = 1.1 * (case.W + 2) * half_ulp_rel
    if not bwd_err <= lowprec_bound:
        return ("WRONG", ulp_err, bwd_err)
    return ("ok" if ulp_err <= 1.5 else "lowprec", ulp_err, bwd_err)
