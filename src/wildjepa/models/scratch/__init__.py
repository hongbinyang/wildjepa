"""Our own I-JEPA implementation.

Public API:
    ScratchEncoder    -- JEPAEncoder-compatible wrapper, for linear-probe/
                         fine-tune/eval use, and as the checkpoint-loading
                         target after pretraining.
    ScratchPredictor  -- JEPAPredictor-compatible wrapper (see base.py for
                         why its contract is looser than JEPAEncoder's).
    IJEPA             -- the composite module (context encoder + EMA target
                         encoder + predictor) that train/pretrain.py actually
                         drives. Not part of the JEPAEncoder/JEPAPredictor
                         interface -- it owns both halves plus the training-
                         time loss, which doesn't fit that narrower contract.
"""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig
from torch import Tensor

from wildjepa.models.base import JEPAEncoder, JEPAPredictor
from wildjepa.models.scratch.ema import update_ema
from wildjepa.models.scratch.loss import jepa_loss
from wildjepa.models.scratch.masking import mask_to_indices
from wildjepa.models.scratch.predictor import Predictor
from wildjepa.models.scratch.vit import VisionTransformer

logger = logging.getLogger(__name__)


class ScratchEncoder(JEPAEncoder):
    """Eval-facing wrapper around a plain VisionTransformer. Used for linear
    probe / fine-tune / cross-backend diffing -- never does I-JEPA masking
    itself (that's only relevant during pretraining, handled by `IJEPA`)."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.vit = VisionTransformer(
            img_size=cfg.encoder.img_size,
            patch_size=cfg.encoder.patch_size,
            embed_dim=cfg.encoder.embed_dim,
            depth=cfg.encoder.depth,
            num_heads=cfg.encoder.num_heads,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.vit.forward_full(x)

    def load_pretrained(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Accept either a raw VisionTransformer state_dict or a full
        # pretraining checkpoint dict with a "context_encoder" key (see
        # train/common.py save_checkpoint).
        if isinstance(state, dict) and "context_encoder" in state:
            state = state["context_encoder"]
        missing, unexpected = self.vit.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("ScratchEncoder.load_pretrained: missing=%s unexpected=%s", missing, unexpected)

    @property
    def embed_dim(self) -> int:
        return self.vit.embed_dim


class ScratchPredictor(JEPAPredictor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.predictor = Predictor(
            encoder_embed_dim=cfg.encoder.embed_dim,
            predictor_embed_dim=cfg.predictor.embed_dim,
            depth=cfg.predictor.depth,
            num_heads=cfg.predictor.num_heads,
            num_patches=(cfg.encoder.img_size // cfg.encoder.patch_size) ** 2,
        )

    def forward(self, *args, **kwargs):
        return self.predictor(*args, **kwargs)


class IJEPA(torch.nn.Module):
    """The composite module train/pretrain.py trains: context encoder
    (trainable), target encoder (EMA-only, no gradient), and predictor.
    `forward` computes the I-JEPA loss for one batch; `update_target_encoder`
    performs the EMA step after the optimizer step.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        vit_kwargs = dict(
            img_size=cfg.encoder.img_size,
            patch_size=cfg.encoder.patch_size,
            embed_dim=cfg.encoder.embed_dim,
            depth=cfg.encoder.depth,
            num_heads=cfg.encoder.num_heads,
        )
        self.context_encoder = VisionTransformer(**vit_kwargs)
        self.target_encoder = VisionTransformer(**vit_kwargs)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(
            encoder_embed_dim=cfg.encoder.embed_dim,
            predictor_embed_dim=cfg.predictor.embed_dim,
            depth=cfg.predictor.depth,
            num_heads=cfg.predictor.num_heads,
            num_patches=self.context_encoder.num_patches,
        )

    def forward(self, images: Tensor, context_mask: Tensor, target_masks: list[Tensor]) -> Tensor:
        context_tokens, context_idx, context_pad = self.context_encoder.forward_masked(images, context_mask)

        with torch.no_grad():
            full_target_tokens = self.target_encoder.forward_full(images)
        D = full_target_tokens.shape[-1]

        target_idx_list = [mask_to_indices(m) for m in target_masks]
        targets = [
            full_target_tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)).detach()
            for idx in target_idx_list
        ]

        predictions = self.predictor(context_tokens, context_idx, context_pad, target_idx_list)
        return jepa_loss(predictions, targets)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        update_ema(self.context_encoder, self.target_encoder, momentum)

    def trainable_parameters(self):
        """Context encoder + predictor only -- the target encoder is EMA-only
        and must never receive a gradient update."""
        yield from self.context_encoder.parameters()
        yield from self.predictor.parameters()
