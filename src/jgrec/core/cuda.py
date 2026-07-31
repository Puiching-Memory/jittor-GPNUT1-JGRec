from __future__ import annotations

from typing import Any


def require_jittor_cuda(jt: Any) -> None:
    """Enable Jittor CUDA or fail before starting a GPU-only workflow."""
    if not bool(jt.has_cuda):
        raise RuntimeError("CUDA is required for this Jittor workflow")
    jt.flags.use_cuda = 1
