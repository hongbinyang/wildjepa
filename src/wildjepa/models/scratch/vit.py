"""A plain ViT encoder, used as the backbone for both the context/target
encoders. Module and parameter names (`patch_embed.proj`, `blocks.N.norm1`,
`blocks.N.attn.{qkv,proj}`, `blocks.N.norm2`, `blocks.N.mlp.{fc1,fc2}`,
`norm`) deliberately mirror facebookresearch/ijepa's `vision_transformer.py`
naming, so an official pretrained checkpoint loads into this class via
`load_state_dict(strict=False)` with minimal key remapping -- that's what
makes the cross-backend correctness check in docs/design.md feasible. (Not
yet verified against an actual downloaded official checkpoint in this
environment -- see docs/design.md limitations.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from wildjepa.models.scratch.masking import gather_with_padding
from wildjepa.models.scratch.patch_embed import PatchEmbed
from wildjepa.models.scratch.pos_embed import get_2d_sincos_pos_embed


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """x: (B, N, C). key_padding_mask: (B, N) bool, True = ignore (padding)."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, N) -- broadcast over query dim
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """Plain ViT. Two forward modes:

    - `forward_full`: every patch, no masking -- used for the target encoder
      (which must see the whole image) and for downstream eval/linear-probe/
      fine-tune (which don't do I-JEPA-style masking at all).
    - `forward_masked`: only patches where `patch_mask` is True, gathered and
      padded to a uniform length within the batch -- used for the context
      encoder during self-supervised pretraining.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0))  # (1, N, D)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward_full(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) -> (B, N, D), every patch, post-norm."""
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_masked(self, x: Tensor, patch_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x: (B, C, H, W). patch_mask: (B, N) bool, True = keep.

        Returns (tokens, idx, pad_mask):
            tokens: (B, num_patches, D) post-norm representations of kept patches
            idx: (B, num_patches) long, original patch-grid index of each token
            pad_mask: (B, num_patches) bool, True = padding (not a real patch)

        Always padded to the full `num_patches` (not each batch's own smaller
        dynamic max) -- see gather_with_padding's pad_to docs for why: a
        varying shape here forces MPS to recompile its graph on every batch.
        """
        x = self.patch_embed(x) + self.pos_embed
        x, idx, pad_mask = gather_with_padding(x, patch_mask, pad_to=self.num_patches)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.norm(x), idx, pad_mask

    # Alias so a VisionTransformer can be used directly wherever a plain
    # `forward(x) -> (B, N, D)` callable is expected (e.g. as the model
    # underneath ScratchEncoder).
    def forward(self, x: Tensor) -> Tensor:
        return self.forward_full(x)
