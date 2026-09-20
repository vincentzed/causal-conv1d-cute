"""Provide optional In-Kernel Event Tracing (IKET) phase markers.

Instrumentation is enabled when the CONV1D_IKET environment variable is set to 1. When disabled,
marker calls are no-ops.

IKET cannot coexist with CUPTI in the same process.
"""

import os

ENABLED = os.environ.get("CONV1D_IKET", "0") == "1"
_iket = None
if ENABLED:
    try:
        from cutlass.cute.experimental import iket as _iket
    except (ImportError, NotImplementedError):
        from cutlass.cute import iket as _iket


def push(name: str) -> None:
    """Push an IKET instrumentation range marker.

    Args:
        name: Label identifying the traced phase.
    """
    if ENABLED:
        _iket.range_push(name)


def pop() -> None:
    """Pop the current IKET instrumentation range marker."""
    if ENABLED:
        _iket.range_pop()
