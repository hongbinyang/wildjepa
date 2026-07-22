"""Device resolution: cuda > mps > cpu, with a single choke point so backends
and training scripts never call torch.cuda/mps checks directly.

Kept deliberately dependency-light (plain torch, no `accelerate`) since the
quick-feasibility-check phase runs single-device. Multi-GPU/TPU/cloud support
extends this module later without touching call sites -- see docs/roadmap.md.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def resolve_device(name: str = "auto", allow_mps_fallback: bool = True) -> torch.device:
    """Resolve a config-specified device name to a concrete torch.device.

    Args:
        name: "auto", "cuda", "mps", or "cpu". "auto" picks the best available.
        allow_mps_fallback: if True, sets PYTORCH_ENABLE_MPS_FALLBACK=1 so
            unsupported ops fall back to CPU instead of raising. MPS-only;
            no-op on other devices.
    """
    if name == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        resolved = name

    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available on this machine.")
    if resolved == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps requested but MPS is not available on this machine.")

    if resolved == "mps" and allow_mps_fallback:
        # Must be set before any MPS op runs; harmless if already set.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    logger.info("Resolved device: %s", resolved)
    return torch.device(resolved)


def device_summary(device: torch.device) -> str:
    """Human-readable one-liner for logging at the start of a run."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"cuda ({name})"
    if device.type == "mps":
        return "mps (Apple Silicon; some ops may fall back to CPU)"
    return "cpu"
