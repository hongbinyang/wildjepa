"""Shared training utilities: checkpointing and a running-average meter."""

from __future__ import annotations

from pathlib import Path

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def save_pretrain_checkpoint(model, path: str | Path) -> None:
    """Saves the pieces a downstream ScratchEncoder/eval script needs:
    the context encoder's weights (what gets fine-tuned/probed downstream)
    plus the predictor (in case pretraining is resumed later)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "context_encoder": model.context_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
        },
        path,
    )


def load_pretrain_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=True)


def save_training_checkpoint(path: str | Path, epoch: int, model, optimizer) -> None:
    """Saves full resumable training state (model + optimizer + epoch index),
    for runs that need to survive being killed and picked back up later --
    unlike save_pretrain_checkpoint above, which only saves the final
    weights a downstream eval script needs, not enough to resume training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_training_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=True)
