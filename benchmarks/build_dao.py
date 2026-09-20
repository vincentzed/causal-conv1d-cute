"""Build the upstream Dao-AILab/causal-conv1d extension for the local GPU.

Uses the nvcc flags of the upstream setup.py, including --use_fast_math, but a single code
generation target instead of the upstream multi-architecture binary. The source tree is expected at
$CONV1D_LAB/refs/causal-conv1d and the extension is written to $CONV1D_LAB/build/dao, where the dao
benchmark provider loads it from. TORCH_CUDA_ARCH_LIST selects the target and defaults to the
capability of device 0.

Usage:
    git clone https://github.com/Dao-AILab/causal-conv1d $CONV1D_LAB/refs/causal-conv1d
    python build_dao.py
"""

import os
from pathlib import Path
import torch
from torch.utils import cpp_extension

LAB = Path(os.environ.get("CONV1D_LAB", str(Path(__file__).resolve().parent)))
SOURCES = (
    "causal_conv1d.cpp",
    "causal_conv1d_fwd.cu",
    "causal_conv1d_bwd.cu",
    "causal_conv1d_update.cu",
)
NVCC_FLAGS = [
    "-O3",
    "-std=c++20",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]


def main():
    """Compile the extension into $CONV1D_LAB/build/dao."""
    major, minor = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    src = LAB / "refs" / "causal-conv1d" / "csrc"
    out = LAB / "build" / "dao"
    out.mkdir(parents=True, exist_ok=True)
    cpp_extension.load(
        name="causal_conv1d_cuda",
        sources=[str(src / f) for f in SOURCES],
        extra_include_paths=[str(src)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=NVCC_FLAGS,
        build_directory=str(out),
        is_python_module=True,
        verbose=True,
    )
    print("built", out)


if __name__ == "__main__":
    main()
