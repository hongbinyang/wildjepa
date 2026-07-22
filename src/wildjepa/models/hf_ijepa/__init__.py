"""Adapter over transformers' I-JEPA model
(https://huggingface.co/docs/transformers/en/model_doc/ijepa). Fastest path
to a working pretrained checkpoint -- no external repo checkout needed.
Used for cross-checking ScratchEncoder's outputs and as a convenience
pretrained baseline; this backend is not wired into the pretraining loop
(train/pretrain.py only trains backend=scratch -- see docs/design.md).

Not exercised end-to-end in this project's own dev sandbox (needs internet
access to fetch model weights from the Hugging Face Hub on first use); code
follows the documented `transformers` API.
"""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig
from torch import Tensor

from wildjepa.models.base import JEPAEncoder
from wildjepa.utils.checkpoint import load_checkpoint_preferring_weights_only

logger = logging.getLogger(__name__)


class HfIJEPAEncoder(JEPAEncoder):
    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "The hf_ijepa backend requires `transformers` (pip install transformers, "
                "already in environment.yml)."
            ) from e

        self.model = AutoModel.from_pretrained(cfg.model_id)
        self._embed_dim = self.model.config.hidden_size
        self._device = device

    def forward(self, x: Tensor) -> Tensor:
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state  # (B, N, D)

    def load_pretrained(self, checkpoint_path: str) -> None:
        state = load_checkpoint_preferring_weights_only(checkpoint_path)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("HfIJEPAEncoder.load_pretrained: missing=%s unexpected=%s", missing, unexpected)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim
