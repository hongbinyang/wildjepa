"""Non-overlapping patch embedding via a strided convolution -- the standard
ViT patchify operation."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) -> (B, num_patches, embed_dim)"""
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            raise ValueError(
                f"Expected {self.img_size}x{self.img_size} input, got {x.shape[-2]}x{x.shape[-1]}"
            )
        x = self.proj(x)  # (B, embed_dim, grid, grid)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
