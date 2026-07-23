"""End-to-end fine-tuning: encoder (unfrozen) + a linear classifier head,
trained jointly with cross-entropy. The upper bound in the evaluation
protocol described in docs/design.md -- linear probe first, this second."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from wildjepa.data import build_dataset, make_supervised_collate_fn
from wildjepa.eval.metrics import accuracy, macro_f1, per_class_f1
from wildjepa.models.base import JEPAEncoder
from wildjepa.train.common import AverageMeter, MetricsLogger

logger = logging.getLogger(__name__)

# See erm_baseline.py for why only these two splits get per-class logging:
# they're the ones actually compared against PUBLISHED_BASELINES.
_PER_CLASS_LOGGED_SPLITS = ("id_test", "test")


class EncoderWithHead(nn.Module):
    def __init__(self, encoder: JEPAEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)  # (B, N, D)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


@torch.no_grad()
def _evaluate(model: EncoderWithHead, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return {
        "macro_f1": macro_f1(labels_np, preds),
        "accuracy": accuracy(labels_np, preds),
        "per_class_f1": per_class_f1(labels_np, preds),
    }


def run_finetune(cfg: DictConfig, device: torch.device, encoder: JEPAEncoder) -> dict[str, float]:
    bundle = build_dataset(cfg.data)
    collate_fn = make_supervised_collate_fn()

    loaders = {
        split: DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn,
        )
        for split, ds in bundle.splits.items()
    }

    model = EncoderWithHead(encoder, bundle.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss()
    metrics_logger = MetricsLogger(cfg.output_dir)

    train_loader = loaders["train"]
    global_step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        meter = AverageMeter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter.update(loss.item(), images.size(0))
            global_step += 1
            metrics_logger.log_scalar("train/loss_step", loss.item(), global_step)

        logger.info("finetune epoch %d/%d avg loss %.4f", epoch + 1, cfg.train.epochs, meter.avg)
        metrics_logger.log_scalar("train/loss_epoch", meter.avg, epoch + 1)

        if (epoch + 1) % cfg.train.eval_every == 0:
            for split, loader in loaders.items():
                if split == "train":
                    continue
                metrics = _evaluate(model, loader, device)
                logger.info("  [%s] macro_f1=%.4f accuracy=%.4f", split, metrics["macro_f1"], metrics["accuracy"])
                metrics_logger.log_scalar(f"{split}/macro_f1", metrics["macro_f1"], epoch + 1)
                metrics_logger.log_scalar(f"{split}/accuracy", metrics["accuracy"], epoch + 1)
                if split in _PER_CLASS_LOGGED_SPLITS:
                    metrics_logger.log_per_class(f"{split}/per_class_f1", metrics["per_class_f1"], epoch + 1)

    results: dict[str, float] = {}
    for split, loader in loaders.items():
        if split == "train":
            continue
        metrics = _evaluate(model, loader, device)
        results[f"{split}_macro_f1"] = metrics["macro_f1"]
        results[f"{split}_accuracy"] = metrics["accuracy"]

    metrics_logger.close()
    return results
