"""Frozen-encoder linear-probe evaluation -- the protocol the original
I-JEPA paper itself uses. Features are extracted once (no gradient), then a
standard scikit-learn LogisticRegression is fit on them; this is standard
practice in the SSL literature and avoids having to hand-roll a second
training loop with its own optimizer/schedule just for a linear head."""

from __future__ import annotations

import logging

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from wildjepa.data import build_dataset, make_supervised_collate_fn
from wildjepa.eval.metrics import accuracy, macro_f1, per_class_f1
from wildjepa.models.base import JEPAEncoder
from wildjepa.train.common import MetricsLogger

logger = logging.getLogger(__name__)

# See erm_baseline.py for why only these two splits get per-class logging.
_PER_CLASS_LOGGED_SPLITS = ("id_test", "test")


@torch.no_grad()
def extract_features(encoder: JEPAEncoder, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    feats, labels = [], []
    for images, batch_labels in loader:
        images = images.to(device)
        tokens = encoder(images)  # (B, N, D)
        pooled = tokens.mean(dim=1)  # mean-pool patch tokens, per the I-JEPA eval protocol
        feats.append(pooled.cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def run_linear_probe(cfg: DictConfig, device: torch.device, encoder: JEPAEncoder) -> dict[str, float]:
    bundle = build_dataset(cfg.data)
    collate_fn = make_supervised_collate_fn()

    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split, ds in bundle.splits.items():
        loader = DataLoader(
            ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=collate_fn
        )
        features[split], labels[split] = extract_features(encoder, loader, device)
        logger.info("Extracted features for split=%s: %s", split, features[split].shape)

    if "train" not in features:
        raise ValueError(f"No 'train' split in dataset splits: {list(features)}")

    clf = LogisticRegression(max_iter=2000)
    clf.fit(features["train"], labels["train"])

    metrics_logger = MetricsLogger(cfg.output_dir)
    results: dict[str, float] = {}
    for split in features:
        if split == "train":
            continue
        preds = clf.predict(features[split])
        split_macro_f1 = macro_f1(labels[split], preds)
        split_accuracy = accuracy(labels[split], preds)
        results[f"{split}_macro_f1"] = split_macro_f1
        results[f"{split}_accuracy"] = split_accuracy

        # Single-point logging (step=0): linear probe is a one-shot fit, not
        # an epoch loop, but still lands in outputs/<run_name>/tensorboard/
        # for side-by-side comparison against other runs.
        metrics_logger.log_scalar(f"{split}/macro_f1", split_macro_f1, 0)
        metrics_logger.log_scalar(f"{split}/accuracy", split_accuracy, 0)
        if split in _PER_CLASS_LOGGED_SPLITS:
            metrics_logger.log_per_class(f"{split}/per_class_f1", per_class_f1(labels[split], preds), 0)

    metrics_logger.close()
    return results
