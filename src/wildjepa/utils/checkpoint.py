"""Shared checkpoint-loading helper for backends that load third-party
checkpoints (fb_ijepa, hf_ijepa) whose pickle contents we don't control --
unlike our own scratch checkpoints (train/common.py, models/scratch), which
always use weights_only=True directly since we know exactly what's in them.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def load_checkpoint_preferring_weights_only(checkpoint_path: str, map_location: str = "cpu"):
    """Tries the safer `weights_only=True` load first (PyTorch's recommended
    default going forward); falls back to `weights_only=False` with a logged
    warning if the checkpoint contains pickled types outside torch's safe
    list (common in older/third-party checkpoints that store things like
    argparse Namespaces alongside tensors)."""
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as e:  # noqa: BLE001 -- torch raises varying error types here
        logger.warning(
            "weights_only=True load failed for %s (%s); falling back to weights_only=False. "
            "Only do this for checkpoints from a source you trust.",
            checkpoint_path,
            e,
        )
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
