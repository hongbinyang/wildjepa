"""Evaluation metrics. Macro-F1 is the primary metric -- it's the official
iWildCam2020-WILDS leaderboard metric specifically because it weights rare
species equally with common ones, unlike raw accuracy (see docs/design.md)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return dict(zip(labels, (float(s) for s in scores), strict=True))


def few_shot_indices(labels: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Indices selecting up to `k` examples per class -- for testing label
    efficiency on rare species specifically, rather than overall accuracy."""
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(labels):
        cls_indices = np.nonzero(labels == cls)[0]
        rng.shuffle(cls_indices)
        selected.append(cls_indices[:k])
    return np.concatenate(selected)
