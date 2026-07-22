"""Adapter over facebookresearch/ijepa (https://github.com/facebookresearch/ijepa).
Reference implementation -- used to validate ScratchEncoder's outputs by
loading the same pretrained weights into both and diffing, and as a strong
pretrained baseline. Not wired into this project's own pretraining loop.

Requires the repo checked out locally (not a pip package):
    git clone https://github.com/facebookresearch/ijepa <repo_path>

Not exercised end-to-end in this project's own dev sandbox -- we don't have
that repo checked out here. Import path and checkpoint-key assumptions below
are based on the repo's public structure (`src/models/vision_transformer.py`
exposing builder functions like `vit_huge`, checkpoints storing encoder
weights under a `"target_encoder"` key); treat as best-effort until verified
against an actual checkout (see docs/design.md limitations).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import Tensor

from wildjepa.models.base import JEPAEncoder
from wildjepa.utils.checkpoint import load_checkpoint_preferring_weights_only

logger = logging.getLogger(__name__)


class FbIJEPAEncoder(JEPAEncoder):
    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super().__init__()
        repo_path = Path(cfg.repo_path).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"fb_ijepa backend requires facebookresearch/ijepa checked out at {repo_path}.\n"
                f"  git clone https://github.com/facebookresearch/ijepa {repo_path}\n"
                "Or point configs/backend/fb_ijepa.yaml's repo_path (or $WILDJEPA_FB_IJEPA_PATH) "
                "somewhere it's already checked out."
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            from src.models import vision_transformer as fb_vit
        except ImportError as e:
            raise ImportError(
                f"Found {repo_path} but could not import src.models.vision_transformer from it. "
                "Verify this is a valid checkout of facebookresearch/ijepa."
            ) from e

        builder = getattr(fb_vit, cfg.arch, None)
        if builder is None:
            available = [n for n in dir(fb_vit) if n.startswith("vit_")]
            raise ValueError(f"Unknown fb_ijepa arch {cfg.arch!r}. Available: {available}")

        self.model = builder()
        self._embed_dim = self.model.embed_dim
        self._device = device

        if cfg.pretrained_checkpoint:
            self.load_pretrained(cfg.pretrained_checkpoint)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def load_pretrained(self, checkpoint_path: str) -> None:
        ckpt = load_checkpoint_preferring_weights_only(checkpoint_path)
        state_dict = ckpt.get("target_encoder", ckpt.get("encoder", ckpt))
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning("FbIJEPAEncoder.load_pretrained: missing=%s unexpected=%s", missing, unexpected)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim
