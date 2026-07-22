"""EMA update for the target encoder, and the momentum schedule that drives it.

I-JEPA's target encoder is never trained by gradient descent -- it's an
exponential moving average of the context encoder's weights, updated after
every optimizer step. The momentum ramps from `start` (more responsive to
the context encoder early in training) to `end` (more stable / slow-moving
late in training), linearly over the run.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def update_ema(context_encoder: nn.Module, target_encoder: nn.Module, momentum: float) -> None:
    """target_param <- momentum * target_param + (1 - momentum) * context_param, in place."""
    for p_ctx, p_tgt in zip(context_encoder.parameters(), target_encoder.parameters(), strict=True):
        p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1 - momentum)


def momentum_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    """Linear ramp from `start` to `end` over `total_steps`. Clamped, so it's
    safe to call with step > total_steps (holds at `end`)."""
    if total_steps <= 0:
        return end
    progress = min(max(step / total_steps, 0.0), 1.0)
    return start + (end - start) * progress
