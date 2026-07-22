"""Phase 0 baseline sanity check: plain supervised ResNet-50 ERM on
iWildCam2020-WILDS, to confirm the eval harness (macro-F1, split handling)
reproduces the published WILDS numbers before trusting anything built on
top of it. See docs/design.md "Evaluation protocol" and docs/roadmap.md
Phase 0.

Independent of the JEPAEncoder backend abstraction -- this is a supervised
baseline, not a JEPA variant, so it doesn't route through
wildjepa.models.base.build_encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from wildjepa.data import build_dataset, make_supervised_collate_fn
from wildjepa.eval.metrics import accuracy, macro_f1
from wildjepa.train.common import (
    AverageMeter,
    load_training_checkpoint,
    save_training_checkpoint,
)

logger = logging.getLogger(__name__)


def _build_resnet50(num_classes: int) -> nn.Module:
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return {"macro_f1": macro_f1(labels_np, preds), "accuracy": accuracy(labels_np, preds)}


def run_erm_baseline(cfg: DictConfig, device: torch.device) -> dict[str, float]:
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

    model = _build_resnet50(bundle.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
    latest_checkpoint = checkpoint_dir / "erm_baseline_latest.pt"
    checkpoint_every = cfg.train.get("checkpoint_every_epochs", 1)

    start_epoch = 0
    resume_from = cfg.train.get("resume_from", None)
    if resume_from:
        state = load_training_checkpoint(resume_from, map_location=str(device))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        logger.info("Resumed from %s: completed epoch %d, continuing at epoch %d", resume_from, state["epoch"] + 1, start_epoch + 1)

    train_loader = loaders["train"]
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        meter = AverageMeter()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter.update(loss.item(), images.size(0))
            if (step + 1) % cfg.train.log_every == 0:
                logger.info("epoch %d step %d/%d loss %.4f", epoch + 1, step + 1, len(train_loader), meter.avg)

        logger.info("erm_baseline epoch %d/%d avg loss %.4f", epoch + 1, cfg.train.epochs, meter.avg)

        if (epoch + 1) % checkpoint_every == 0:
            epoch_checkpoint = checkpoint_dir / f"erm_baseline_epoch{epoch + 1}.pt"
            save_training_checkpoint(epoch_checkpoint, epoch, model, optimizer)
            save_training_checkpoint(latest_checkpoint, epoch, model, optimizer)
            logger.info("Saved checkpoint: %s", epoch_checkpoint)

        if (epoch + 1) % cfg.train.eval_every == 0:
            for split, loader in loaders.items():
                if split == "train":
                    continue
                metrics = _evaluate(model, loader, device)
                logger.info("  [%s] macro_f1=%.4f accuracy=%.4f", split, metrics["macro_f1"], metrics["accuracy"])

    results: dict[str, float] = {}
    for split, loader in loaders.items():
        if split == "train":
            continue
        metrics = _evaluate(model, loader, device)
        results[f"{split}_macro_f1"] = metrics["macro_f1"]
        results[f"{split}_accuracy"] = metrics["accuracy"]

    return results
