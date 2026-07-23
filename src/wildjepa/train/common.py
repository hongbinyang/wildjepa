"""Shared training utilities: checkpointing, a running-average meter, and
TensorBoard metrics logging."""

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


class MetricsLogger:
    """Thin wrapper around torch.utils.tensorboard.SummaryWriter.

    Writes to <output_dir>/tensorboard/, so a run's directory (identified by
    run_name, see docs/lifecycle.md "Run identity") holds its metrics
    alongside its checkpoints, and `tensorboard --logdir outputs/` picks up
    every run_name automatically for side-by-side comparison -- no separate
    tracking service or account needed, unlike e.g. Weights & Biases (listed
    as an optional, unwired dependency in environment.yml).

    Import of SummaryWriter is deferred into __init__ so importing this
    module doesn't require the tensorboard package unless a logger is
    actually constructed.
    """

    def __init__(self, output_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(Path(output_dir) / "tensorboard"))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def log_per_class(self, tag_prefix: str, per_class: dict[int, float], step: int) -> None:
        """Logs one scalar per class under f"{tag_prefix}/class_<id>" --
        TensorBoard's own UI groups these back into one comparable chart.
        Not printed to the console logs (182 lines per split per epoch would
        drown everything else out); this is specifically the rare-species
        visibility the console-only macro-F1 number doesn't give you."""
        for class_id, value in per_class.items():
            self._writer.add_scalar(f"{tag_prefix}/class_{class_id}", value, step)

    def close(self) -> None:
        self._writer.close()
