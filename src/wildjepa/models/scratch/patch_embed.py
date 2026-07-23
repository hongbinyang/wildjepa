"""Non-overlapping patch embedding -- the standard ViT patchify operation."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
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
        # Kept as a real nn.Conv2d (not e.g. an equivalent nn.Linear) purely
        # for its state_dict shape/key ("proj.weight"/"proj.bias") -- that's
        # what makes an official facebookresearch/ijepa checkpoint's
        # patch_embed.proj load into this module directly (see vit.py).
        # forward() below does NOT call this module directly, though --
        # see there for why.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) -> (B, num_patches, embed_dim)

        Computed via unfold + linear, mathematically identical to
        `self.proj(x).flatten(2).transpose(1, 2)` (verified numerically
        equal to float32 precision) but deliberately not calling
        `self.proj(x)` (a strided Conv2d) directly: on Apple Silicon MPS,
        Conv2d's backward raises `RuntimeError: view size is not compatible
        with input tensor's size and stride` whenever its output feeds a
        further op after a transpose/permute -- exactly what every caller of
        this class does immediately afterward (`+ pos_embed`, masking,
        attention). Reproduced and isolated outside this codebase (a bare
        `nn.Conv2d` -> `.transpose` -> `nn.Linear` backward fails the same
        way); this is a PyTorch/MPS interaction bug, not anything specific
        to this model. `F.unfold` isn't MPS-native either and falls back to
        CPU (a one-time `im2col` warning, harmless -- see
        `utils/device.py`'s `PYTORCH_ENABLE_MPS_FALLBACK`), but its backward
        doesn't have this bug.
        """
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            raise ValueError(
                f"Expected {self.img_size}x{self.img_size} input, got {x.shape[-2]}x{x.shape[-1]}"
            )
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)  # (B, C*p*p, N)
        patches = patches.transpose(1, 2)  # (B, N, C*p*p)
        weight = self.proj.weight.reshape(self.proj.out_channels, -1)  # (D, C*p*p)
        return F.linear(patches, weight, self.proj.bias)  # (B, N, D)
