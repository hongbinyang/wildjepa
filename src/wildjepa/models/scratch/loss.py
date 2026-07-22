"""The I-JEPA training objective: average smooth-L1 (Huber) distance between
predicted and true target-block representations, one term per target block.

Smooth L1 (rather than plain MSE) is the official implementation's choice --
it's less sensitive to the occasional large-magnitude representation than raw
L2, which matters since target representations are unconstrained (unlike,
say, softmax outputs)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def jepa_loss(predictions: list[Tensor], targets: list[Tensor]) -> Tensor:
    """predictions, targets: same-length lists of (B, K, D) tensors, one per
    target block. `targets` must already be detached (no grad should flow
    into the target encoder through this loss -- see IJEPA.forward)."""
    if len(predictions) != len(targets):
        raise ValueError(f"Got {len(predictions)} predictions but {len(targets)} targets")
    losses = [F.smooth_l1_loss(p, t) for p, t in zip(predictions, targets, strict=True)]
    return torch.stack(losses).mean()
