"""Define convolution configurations and test cases for causal conv1d.

Configurations define (dim, width, bias, activation), where dim is the channel count, width is the
kernel width, bias indicates whether a bias vector is used, and activation is the post-convolution
activation function.

Supported layouts: btd: (batch, seqlen, dim) bdl: (batch, dim, seqlen)
"""

from harness import Case

MODELS = {
    "lfm2-1.2b": (2048, 3, False, None, "LiquidAI/LFM2-1.2B,2.6B,8B-A1B,24B-A2B hidden=2048 (H)"),
    "lfm2-350m": (1024, 3, False, None, "LiquidAI/LFM2-350M, LFM2.5-350M hidden=1024 (H)"),
    "nemotron3-nano-30b": (
        6144,
        4,
        True,
        "silu",
        "Nemotron-3-Nano-30B-A3B mamba2 4096+2*8*128 TP1 (H)",
    ),
    "nemotron3-super-tp4": (
        2560,
        4,
        True,
        "silu",
        "Nemotron-3-Super-120B-A12B mamba2 10240/TP4 (H)",
    ),
    "glm5.3-flash-kda-tp4": (
        2048,
        4,
        False,
        "silu",
        "zai-org/GLM-5.3-Flash KDA 64*128/TP4 per q/k/v conv (H)",
    ),
    "kimi-k3-kda-tp8": (
        1536,
        4,
        False,
        "silu",
        "moonshotai/Kimi-K3 KDA 96*128/TP8 per conv (H); = Qwen3.5-397B GDN TP8",
    ),
    "qwen3.5-9b-gdn": (8192, 4, False, "silu", "Qwen3.5-4B/9B/35B-A3B GDN 2*16*128+32*128 TP1 (H)"),
    "qwen3.8-27b-gdn": (
        10240,
        4,
        False,
        "silu",
        "Qwen/Qwen3.8-27B, Qwen3.6-27B GDN 2*16*128+48*128 TP1 (H), recorded from a live server; "
        "also Qwen/Qwen3.8-Flash-Next GDN TP1",
    ),
    "qwen3.8-2.4t-gdn-tp8": (
        2560,
        4,
        False,
        "silu",
        "Qwen/Qwen3.8-2.4T-A95B GDN (2*16*128+128*128)/TP8, 69 of 92 layers; = Qwen3.8-Flash-Next TP4",
    ),
}
DECODE_ONLY = {
    "glm5.3-flash-kda-tp4-fused": (
        6144,
        4,
        False,
        "silu",
        "GLM-5.3-Flash KDA fused qkv decode 3*8192/TP4",
    ),
    "kimi-k3-kda-tp8-fused": (4608, 4, False, "silu", "Kimi-K3 KDA fused qkv decode 3*12288/TP8"),
}
PREFILL_BL = [(1, 512), (1, 2048), (1, 8192), (8, 2048), (32, 512), (1, 2047)]
DECODE_B = [1, 8, 64, 256]


def cases(
    models=None, layouts=("btd", "bdl"), kinds=("fwd", "update"), prefill_bl=None, decode_b=None
):
    """Generate test cases for forward and update convolution kernels.

    Args:
        models: Optional collection of model names to generate cases for. If None, all configured
            models are included.
        layouts: Sequence of memory layout strings for forward cases, such as "btd" or "bdl".
            Defaults to ("btd", "bdl").
        kinds: Sequence of kernel types to include, containing "fwd", "update", or both. Defaults to
            ("fwd", "update").
        prefill_bl: Sequence of (batch, seqlen) pairs for forward cases. If None, uses default
            prefill dimensions.
        decode_b: Sequence of batch sizes for update cases with sequence length 1. If None, uses
            default decode batch sizes.

    Returns:
        List of Case objects for the specified configurations.
    """
    out = []
    for name, (D, W, bias, act, _) in MODELS.items():
        if models and name not in models:
            continue
        if "fwd" in kinds:
            for layout in layouts:
                for B, L in prefill_bl or PREFILL_BL:
                    out.append(Case(name, "fwd", D, W, B, L, bias, act, layout))
        if "update" in kinds:
            for B in decode_b or DECODE_B:
                out.append(Case(name, "update", D, W, B, 1, bias, act))
    if "update" in kinds:
        for name, (D, W, bias, act, _) in DECODE_ONLY.items():
            if models and name not in models:
                continue
            for B in decode_b or DECODE_B:
                out.append(Case(name, "update", D, W, B, 1, bias, act))
    return out
