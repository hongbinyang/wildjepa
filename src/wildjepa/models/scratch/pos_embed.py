"""Fixed (non-learnable) 2D sin-cos positional embeddings, as used in
MAE/DINO/I-JEPA-style ViTs. Pure numpy, converted to a torch buffer by the
caller -- keeps this module dependency-light and independently testable.
"""

from __future__ import annotations

import numpy as np


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Returns (grid_size * grid_size, embed_dim) sin-cos position embeddings.

    embed_dim must be divisible by 4 (split in half for the two spatial axes,
    then each half split again for sin/cos).
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # each (grid_size, grid_size)
    grid = np.stack(grid, axis=0)  # (2, grid_size, grid_size)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (grid_size*grid_size, embed_dim)
    return pos_embed


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (embed_dim/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, embed_dim/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1).astype(np.float32)  # (M, embed_dim)
