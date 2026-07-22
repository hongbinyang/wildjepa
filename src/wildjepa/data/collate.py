"""Collate functions bridging plain (image, label) datasets to what the
pretraining loop (needs masks) vs. supervised loops (need stacked labels)
each expect.

Implemented as callable classes rather than closures: a DataLoader with
num_workers > 0 pickles its collate_fn to hand off to worker processes
(spawn is the default multiprocessing start method on macOS/Windows), and
Python's pickle can't serialize a function object defined inside another
function. A module-level class with __call__ has no such restriction."""

from __future__ import annotations

import torch

from wildjepa.models.scratch.masking import MaskingConfig, MultiBlockMaskCollator


class _PretrainCollate:
    """Batches images and attaches I-JEPA context/target masks. Labels (if
    present in the underlying dataset) are dropped -- pretraining is
    label-free by design."""

    def __init__(self, masking_cfg: MaskingConfig) -> None:
        self.collator = MultiBlockMaskCollator(masking_cfg)

    def __call__(self, batch: list[tuple[torch.Tensor, int]]) -> dict:
        images = torch.stack([item[0] for item in batch])
        return self.collator(images)


def make_pretrain_collate_fn(masking_cfg: MaskingConfig):
    return _PretrainCollate(masking_cfg)


class _SupervisedCollate:
    """Plain (images, labels) batching, for linear-probe feature extraction
    and fine-tuning."""

    def __call__(self, batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return images, labels


def make_supervised_collate_fn():
    return _SupervisedCollate()
