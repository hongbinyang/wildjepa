"""The I-JEPA predictor: takes context-encoder tokens (with their true
positional embeddings) plus a learned mask token (repeated, with the *target*
block's positional embeddings) for each target block, runs them through a
narrow transformer, and reads off the mask-token outputs as the predicted
target representations.

This mirrors the official I-JEPA predictor design (and the MAE decoder it's
descended from): the predictor never sees target-block content, only where
it is asked to predict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from wildjepa.models.scratch.pos_embed import get_2d_sincos_pos_embed
from wildjepa.models.scratch.vit import Block


class Predictor(nn.Module):
    def __init__(
        self,
        encoder_embed_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
        num_patches: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        grid_size = int(round(num_patches**0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"num_patches ({num_patches}) is not a perfect square")
        pos_embed = get_2d_sincos_pos_embed(predictor_embed_dim, grid_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0))  # (1, N, D)

        self.blocks = nn.ModuleList([Block(predictor_embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.proj_back = nn.Linear(predictor_embed_dim, encoder_embed_dim)

    def _pos_at(self, idx: Tensor) -> Tensor:
        """idx: (B, K) long patch indices -> (B, K, D) positional embeddings."""
        return self.pos_embed[0][idx]

    def forward(
        self,
        context_tokens: Tensor,
        context_idx: Tensor,
        context_pad_mask: Tensor,
        target_idx_list: list[Tensor],
    ) -> list[Tensor]:
        """
        context_tokens: (B, Kc, Denc) -- context encoder output (post-norm)
        context_idx: (B, Kc) long -- original patch-grid index of each context token
        context_pad_mask: (B, Kc) bool, True = padding
        target_idx_list: list of (B, Kt) long -- patch-grid indices for each
                          target block (Kt is fixed across the batch and
                          across blocks, by construction of the masking collator)

        Returns: list of (B, Kt, Denc), one predicted representation per target block.
        """
        B = context_tokens.shape[0]
        ctx = self.embed(context_tokens) + self._pos_at(context_idx)

        predictions = []
        for target_idx in target_idx_list:
            Kt = target_idx.shape[1]
            mask_tok = self.mask_token.expand(B, Kt, -1) + self._pos_at(target_idx)

            x = torch.cat([ctx, mask_tok], dim=1)
            pad = torch.cat([context_pad_mask, torch.zeros(B, Kt, dtype=torch.bool, device=x.device)], dim=1)

            for blk in self.blocks:
                x = blk(x, pad)
            x = self.norm(x)

            pred = self.proj_back(x[:, -Kt:])  # mask-token outputs only
            predictions.append(pred)

        return predictions
