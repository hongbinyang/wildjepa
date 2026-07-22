"""Backend-agnostic interfaces for JEPA encoders and predictors.

Three backends implement these interfaces:
  - wildjepa.models.scratch    our own implementation (primary path)
  - wildjepa.models.fb_ijepa   adapter over facebookresearch/ijepa
  - wildjepa.models.hf_ijepa   adapter over transformers' I-JEPA model

Both interfaces extend nn.Module (not just ABC) so every backend gets
parameter registration, `.to(device)`, `.state_dict()`, and optimizer
compatibility for free, regardless of which library it wraps internally.

Keeping JEPAEncoder's interface this narrow is deliberate: it's exactly what
linear-probe/fine-tune evaluation needs (a `forward` that returns patch
representations), and it's also what makes the cross-backend correctness
check possible -- load the same pretrained weights into `scratch` and
`fb_ijepa`/`hf_ijepa`, run the same image through `.forward()` on each, and
diff the output embeddings (see tests/test_cross_backend.py and
docs/design.md).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)


class JEPAEncoder(nn.Module, ABC):
    """Produces patch-level representations for an image batch."""

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) -> patch representations (B, N, D)."""
        raise NotImplementedError

    @abstractmethod
    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load weights from a checkpoint file, in whatever format this
        backend's upstream source uses. Should tolerate partial/mismatched
        keys (strict=False) and log what didn't match, since checkpoints
        moving between backends is the whole point of this abstraction."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        raise NotImplementedError


class JEPAPredictor(nn.Module, ABC):
    """Predicts target-block representations from context-block representations.

    Only the `scratch` backend implements this -- it's the only backend this
    project runs the self-supervised pretraining objective with. Reference
    backends (`fb_ijepa`, `hf_ijepa`) are used for inference/eval/correctness
    checks, not for running our own training loop, so `build_predictor`
    returns None for them.

    The exact call signature is intentionally left to the implementation
    (concrete signature documented in wildjepa.models.scratch.predictor) --
    unlike JEPAEncoder, there's no cross-backend contract to keep narrow here.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def build_encoder(cfg: DictConfig, device: torch.device) -> JEPAEncoder:
    """Factory: cfg.backend.name in {"scratch", "fb_ijepa", "hf_ijepa"}.

    Builds the encoder, moves it to `device`, and loads
    cfg.backend.pretrained_checkpoint if one is set -- centralized here so
    all three backends get consistent checkpoint-loading behavior.
    """
    name = cfg.backend.name

    if name == "scratch":
        from wildjepa.models.scratch import ScratchEncoder

        encoder: JEPAEncoder = ScratchEncoder(cfg.backend)
    elif name == "fb_ijepa":
        from wildjepa.models.fb_ijepa import FbIJEPAEncoder

        encoder = FbIJEPAEncoder(cfg.backend, device)
    elif name == "hf_ijepa":
        from wildjepa.models.hf_ijepa import HfIJEPAEncoder

        encoder = HfIJEPAEncoder(cfg.backend, device)
    else:
        raise ValueError(f"Unknown backend: {name!r}")

    encoder.to(device)

    checkpoint = getattr(cfg.backend, "pretrained_checkpoint", None)
    if checkpoint:
        logger.info("Loading pretrained checkpoint for %s backend: %s", name, checkpoint)
        encoder.load_pretrained(checkpoint)

    return encoder


def build_predictor(cfg: DictConfig, device: torch.device) -> JEPAPredictor | None:
    """Only the scratch backend currently implements a predictor. Returns
    None for reference backends used purely for inference/eval."""
    name = cfg.backend.name

    if name == "scratch":
        from wildjepa.models.scratch import ScratchPredictor

        predictor = ScratchPredictor(cfg.backend)
        predictor.to(device)
        return predictor

    return None
