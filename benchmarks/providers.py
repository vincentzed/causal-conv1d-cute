"""Competing causal-conv1d implementations behind a unified benchmarking interface.

Provides forward and recurrent update execution wrappers for multiple CUDA, Triton, and reference
backends. Benchmark harnesses call provider.fwd(case, t) to obtain a (fn, get_out) pair, or
provider.update(case, t) to obtain a (fn, get_out, get_state) tuple. Zero-argument callable fn is
captured into a CUDA graph with preallocated outputs where supported. get_out returns the (B, D, L)
or (B, D) output of the last invocation for numerical gating, and provider.skip_reason records why
unsupported configurations returned None.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

from causal_conv1d_cute import from_ring, to_ring
from causal_conv1d_cute.api import CausalConv1d

LAB = os.environ.get("CONV1D_LAB", str(Path(__file__).resolve().parent))


class Provider:
    """Base class for causal-conv1d implementation benchmark providers."""

    name = "?"
    graph = True

    def __init__(self):
        """Initialize provider skip reason and execution metadata."""
        self.skip_reason = ""
        self.note = ""

    def _skip(self, why):
        """Record the reason a benchmark case cannot run and return None."""
        self.skip_reason = why
        return None

    def fwd(self, case, t):
        """Prepare forward convolution callable and output tensor getter.

        Args:
            case: Benchmark case parameters including dimensions, layout, and activation.
            t: Dictionary of input tensors containing x, w, and optional bias b.

        Returns:
            Tuple of (fn, get_out) where fn is a zero-argument callable for CUDA graph
            capture and get_out retrieves the output tensor, or None if unsupported.
        """
        return self._skip("no fwd")

    def update(self, case, t):
        """Prepare single-step recurrent update callable and tensor accessors.

        Args:
            case: Benchmark case parameters including dimensions and activation.
            t: Dictionary of input tensors containing x, state, w, and optional bias b.

        Returns:
            Tuple of (fn, get_out, get_state) where fn executes the step, get_out
            retrieves the output tensor, and get_state retrieves the updated state, or
            None if unsupported.
        """
        return self._skip("no update")


class Dao(Provider):
    """Upstream Dao-AILab/causal-conv1d 1.7.0 implementation supporting filter widths 2 through 4."""

    name = "dao"

    def __init__(self):
        """Load upstream Dao-AILab causal_conv1d_cuda extension from the build tree."""
        super().__init__()
        sys.path.insert(0, f"{LAB}/build/dao")
        import causal_conv1d_cuda

        self.ext = causal_conv1d_cuda

    def fwd(self, case, t):
        """Prepare forward convolution restricted to filter widths 2 through 4.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing activations x, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out) calling causal_conv1d_fwd, or None if filter width
            is not in [2, 4].
        """
        if not 2 <= case.W <= 4:
            return self._skip("widths 2-4 only")
        x, w, b = (t["x"], t["w"], t["b"])
        out = torch.empty_like(x)
        silu = case.act == "silu"
        return (
            lambda: self.ext.causal_conv1d_fwd(x, w, b, None, None, out, None, silu),
            lambda: out,
        )

    def update(self, case, t):
        """Prepare recurrent update execution restricted to filter widths 2 through 4.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing input x, recurrent state, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out, get_state) calling causal_conv1d_update, or None if
            filter width is not in [2, 4].
        """
        if not 2 <= case.W <= 4:
            return self._skip("widths 2-4 only")
        x, w, b = (t["x"].unsqueeze(-1), t["w"], t["b"])
        state = t["state"].clone()
        out = torch.empty_like(x)
        silu = case.act == "silu"
        fn = lambda: self.ext.causal_conv1d_update(x, state, w, b, out, silu, None, None)
        return (fn, lambda: out.squeeze(-1), lambda: state)


class Fla(Provider):
    """flash-linear-attention Triton convolution provider operating natively on [B, T, D] layout."""

    name = "fla_triton"

    def __init__(self):
        """Import flash-linear-attention Triton convolution operations."""
        super().__init__()
        from fla.modules.conv.triton import ops

        self.ops = ops

    def fwd(self, case, t):
        """Prepare forward Triton convolution with contiguous [B, T, D] activations.

        For bdl layout cases, the activation tensor is transposed and copied to contiguous [B, T, D]
        storage outside the timed region because the native FLA module always maintains contiguous
        [B, T, D] activations.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing activations x, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out) executing Triton causal_conv1d_fwd.
        """
        x, w, b = (t["x"], t["w"], t["b"])
        xt = x.transpose(1, 2)
        if case.layout == "bdl":
            xt = xt.contiguous()
            self.note = "input pre-copied to [B,T,D] outside the timed region"
        box = {}

        def fn():
            box["y"] = self.ops.causal_conv1d_fwd(
                x=xt,
                weight=w,
                bias=b,
                residual=None,
                initial_state=None,
                output_final_state=False,
                activation=case.act,
            )[0]

        return (fn, lambda: box["y"].transpose(1, 2))

    def update(self, case, t):
        """Prepare update Triton convolution using a width-W input cache.

        FLA maintains a cache of the last W inputs of shape [N, D, W] rather than a W - 1 recurrent
        hidden state, requiring the state to be padded by prepending zeros.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing input x, recurrent state, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out, get_state) executing Triton causal_conv1d_update.
        """
        x, w, b = (t["x"], t["w"], t["b"])
        cache = torch.cat([torch.zeros_like(t["state"][..., :1]), t["state"]], dim=-1).contiguous()
        box = {}

        def fn():
            box["y"] = self.ops.causal_conv1d_update(
                x=x, cache=cache, residual=None, weight=w, bias=b, activation=case.act
            )[0]

        return (fn, lambda: box["y"].reshape(case.B, case.D), lambda: cache[..., 1:])


class CudnnOps(Provider):
    """cuDNN causal_conv1d provider using native CuTe-DSL or generic NVRTC backend kernels."""

    name = "cudnn"

    def __init__(self):
        """Initialize cuDNN causal conv1d operators and query backend versions."""
        super().__init__()
        import cudnn
        from cudnn.ops import causal_conv1d, causal_conv1d_update
        from cudnn.ops.causal_conv1d import _get_causal_conv1d_last_route

        self.f, self.upd, self.route = (
            causal_conv1d,
            causal_conv1d_update,
            _get_causal_conv1d_last_route,
        )
        self.version = f"fe {cudnn.__version__} be {cudnn.backend_version()}"

    def fwd(self, case, t):
        """Execute cuDNN forward convolution and identify the selected backend route.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing activations x, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out) executing cudnn.ops.causal_conv1d, or None if execution
            fails.
        """
        x, w, b = (t["x"], t["w"], t["b"])
        box = {}

        def fn():
            box["y"] = self.f(x, w, b, activation=case.act)

        try:
            fn()
            box["route"] = self.route()
        except Exception as e:
            return self._skip(f"{type(e).__name__}: {str(e)[:120]}")
        self.note = f"route={box['route']}"
        return (fn, lambda: box["y"])

    def update(self, case, t):
        """Execute cuDNN single-step update after verifying support with a probe run.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing input x, recurrent state, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out, get_state) executing cudnn.ops.causal_conv1d_update,
            or None if execution fails.
        """
        x, w, b = (t["x"], t["w"], t["b"])
        state = t["state"].clone()
        box = {}

        def fn():
            box["y"] = self.upd(x, state, w, b, activation=case.act)

        try:
            probe_state = state.clone()
            self.upd(x, probe_state, w, b, activation=case.act)
        except Exception as e:
            return self._skip(f"{type(e).__name__}: {str(e)[:120]}")
        return (fn, lambda: box["y"], lambda: state)


class CudnnState4(CudnnOps):
    """cuDNN decode provider padded to 4-element state for native fast-path execution.

    The native kernel fast path requires a 4-element state to perform a single 64-bit load per
    channel, whereas standard W - 1 = 3 state falls back to a slower scalar-state path. Serves as a
    padded-state baseline against cute_pad for filter width 4.
    """

    name = "cudnn_st4"

    def fwd(self, case, t):
        """Skip forward pass for decode-only provider.

        Args:
            case: Benchmark case configuration.
            t: Dictionary of input tensors.

        Returns:
            None because this provider only supports recurrent update decoding.
        """
        return self._skip("decode-only variant")

    def update(self, case, t):
        """Prepare cuDNN decode update with recurrent state padded to 4 elements.

        Prepends random history to construct a contiguous 4-element state tensor, matching the
        native kernel fast path expectation for width 4.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing input x, recurrent state, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out, get_state) executing cudnn.ops.causal_conv1d_update,
            or None if filter width is not 4 or execution fails.
        """
        if case.W != 4:
            return self._skip("native update is width 4 only")
        x, w, b = (t["x"], t["w"], t["b"])
        g = torch.Generator(device="cuda").manual_seed(7)
        hist = torch.randn(case.B, case.D, 1, device="cuda", dtype=case.dtype, generator=g)
        state = torch.cat([hist, t["state"]], dim=-1).contiguous()
        box = {}

        def fn():
            box["y"] = self.upd(x, state, w, b, activation=case.act)

        try:
            self.upd(x, state.clone(), w, b, activation=case.act)
        except Exception as e:
            return self._skip(f"{type(e).__name__}: {str(e)[:120]}")
        return (fn, lambda: box["y"], lambda: state[..., 1:])


class CudnnNwh(Provider):
    """Provide cuDNN causal conv with [B, L, D] storage and [K, D] weights.

    cuDNN generic backend causal convolution provider requiring [B, L, D] storage and [K, D]
    weights.
    """

    name = "cudnn_nwh"

    def __init__(self):
        """Initialize the cuDNN NWH causal convolution operator."""
        super().__init__()
        from cudnn.ops import causal_conv1d_nwh

        self.f = causal_conv1d_nwh

    def fwd(self, case, t):
        """Prepare forward convolution for contiguous BTD layout with transposed weights.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing activations x, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out) executing cudnn.ops.causal_conv1d_nwh, or None if
            layout is not btd or execution fails.
        """
        if case.layout != "btd":
            return self._skip("NWH storage only")
        x, w, b = (t["x"].transpose(1, 2), t["w"].t().contiguous(), t["b"])
        assert x.is_contiguous()
        act = "silu" if case.act else "identity"
        box = {}

        def fn():
            box["y"] = self.f(x, w, b, activation=act)

        try:
            fn()
        except Exception as e:
            return self._skip(f"{type(e).__name__}: {str(e)[:120]}")
        return (fn, lambda: box["y"].transpose(1, 2))


class SubQ(Provider):
    """Provide NVIDIA subquadratic-ops kernels under channel-last layout.

    NVIDIA subquadratic-ops-torch provider dispatching to native kernels under channel-last layout.
    """

    name = "subq_ops"

    def __init__(self):
        """Initialize NVIDIA subquadratic-ops-torch causal convolution operator."""
        super().__init__()
        from subquadratic_ops_torch.causal_conv1d import causal_conv1d

        self.f = causal_conv1d

    def fwd(self, case, t):
        """Prepare forward convolution setting channel_last flag when layout is BTD.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing activations x, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out) executing causal_conv1d, or None if execution fails.
        """
        w, b = (t["w"], t["b"])
        cl = case.layout == "btd"
        x = t["x"].transpose(1, 2) if cl else t["x"]
        box = {}

        def fn():
            box["y"] = self.f(x, w, b, case.act or "identity", channel_last=cl)

        try:
            fn()
        except Exception as e:
            return self._skip(f"{type(e).__name__}: {str(e)[:120]}")
        return (fn, lambda: box["y"].transpose(1, 2) if cl else box["y"])


class TorchConv(Provider):
    """Reference implementation using grouped F.conv1d and float32 accumulation."""

    name = "torch_conv1d"

    def fwd(self, case, t):
        """Baseline forward convolution using grouped F.conv1d, left-padding, and slicing.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing activations x, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out) computing grouped convolution.
        """
        x, w, b = (t["x"], t["w"].unsqueeze(1), t["b"])
        box = {}

        def fn():
            y = F.conv1d(x, w, b, padding=case.W - 1, groups=case.D)[..., : case.L]
            box["y"] = F.silu(y) if case.act else y

        return (fn, lambda: box["y"])

    def update(self, case, t):
        """Baseline recurrent update concatenating input state and accumulating in float32.

        Accumulation is performed in float32 precision matching the numerical behavior of
        causal_conv1d_update_ref when weights are float32.

        Args:
            case: Benchmark case configuration.
            t: Dictionary containing input x, recurrent state, weights w, and bias b.

        Returns:
            Tuple of (fn, get_out, get_state) updating state in place and returning step output.
        """
        x, w, b = (t["x"], t["w"], t["b"])
        state = t["state"].clone()
        box = {}
        w32 = w.float()
        b32 = b.float() if b is not None else None

        def fn():
            full = torch.cat([state, x.unsqueeze(-1)], dim=-1)
            y = (full.float() * w32).sum(-1)
            if b32 is not None:
                y = y + b32
            box["y"] = (F.silu(y) if case.act else y).to(x.dtype)
            state.copy_(full[..., 1:])

        return (fn, lambda: box["y"], lambda: state)


class Cute(Provider):
    """Provider wrapping CuTe DSL causal convolution kernels.

    Supports the standard conv state, the padded state and the ring state for decode operations.
    The ring variant converts the reference state with to_ring, decodes on it, and hands the
    harness the state converted back, so the same exact-equality check applies.
    """

    def __init__(self, name):
        """Initialize the Cute provider.

        Args:
            name: Name of the provider variant: cute, cute_pad or cute_ring.
        """
        super().__init__()
        self.name, self.padded, self.force = (name, name == "cute_pad", None)

    def fwd(self, case, t):
        """Set up forward convolution execution and output allocation.

        Allocates the output tensor according to the case layout, preserving transposed memory
        layout for non-bdl formats, and returns execution closures.

        Args:
            case: Case specification containing layout and act attributes.
            t: Dictionary of input tensors containing w, b, and x.

        Returns:
            A tuple of (run_func, out_func) where run_func executes the forward pass
            and out_func returns the output tensor.
        """
        layer = CausalConv1d(t["w"], t["b"], case.act)
        x = t["x"]
        out = (
            torch.empty_like(x)
            if case.layout == "bdl"
            else torch.empty_like(x.transpose(1, 2)).transpose(1, 2)
        )
        cfg = self.force
        return (lambda: layer.fwd(x, out=out, cfg=cfg), lambda: out)

    def update(self, case, t):
        """Set up single-step state update convolution execution.

        If cute_pad is configured and the padded filter width P exceeds W - 1, allocates a padded
        state of size (B, D, P) with random historical prefix data and places the active state in
        the trailing W - 1 entries. Otherwise, operates on a clone of the standard state.

        Args:
            case: Case specification containing W, B, D, dtype, and act attributes.
            t: Dictionary of input tensors containing w, b, x, and state.

        Returns:
            A tuple of (run_func, out_func, view_func) where run_func executes the layer
            update, out_func returns the output tensor, and view_func returns a slice
            corresponding to the standard (W - 1) state entries.
        """
        layer = CausalConv1d(t["w"], t["b"], case.act)
        x, out = (t["x"], torch.empty_like(t["x"]))
        W, P = (case.W, layer.wpad.shape[1])
        cfg = self.force
        if self.name == "cute_ring":
            seen = torch.full((case.B,), 7, device="cuda", dtype=torch.int32)
            ring = to_ring(t["state"], seen).contiguous()
            run = lambda: layer.update_ring(x, ring, seen, out=out, cfg=cfg)
            return (run, lambda: out, lambda: from_ring(ring, seen + 1))
        if self.padded and P != W - 1:
            g = torch.Generator(device="cuda").manual_seed(7)
            hist = torch.randn(
                case.B, case.D, P - (W - 1), device="cuda", dtype=case.dtype, generator=g
            )
            state = torch.cat([hist, t["state"]], dim=-1).contiguous()
            view = lambda: state[..., P - (W - 1) :]
        else:
            state = t["state"].clone()
            view = lambda: state
        return (lambda: layer.update(x, state, out=out, cfg=cfg), lambda: out, view)


def load(names):
    """Instantiate benchmark providers by name and collect import failures.

    Args:
        names: Sequence of provider name strings to instantiate.

    Returns:
        Tuple of (out, failed), where out is a list of instantiated Provider instances
        and failed is a dictionary mapping provider names to failure error strings.
    """
    table = {
        "dao": Dao,
        "fla_triton": Fla,
        "cudnn": CudnnOps,
        "cudnn_st4": CudnnState4,
        "cudnn_nwh": CudnnNwh,
        "subq_ops": SubQ,
        "torch_conv1d": TorchConv,
    }
    out, failed = ([], {})
    for n in names:
        try:
            if n.startswith("cute"):
                out.append(Cute(n))
            else:
                out.append(table[n]())
        except Exception as e:
            failed[n] = f"{type(e).__name__}: {str(e)[:200]}"
    return (out, failed)
