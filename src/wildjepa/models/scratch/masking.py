"""Multi-block masking for I-JEPA (Assran et al., 2023, arXiv:2301.08243, Sec 3.2).

Per image: sample one large "context" block and `num_target_blocks` smaller
"target" blocks on the patch grid. Remove any patches that fall inside a
target block from the context block, so the context encoder never sees what
the predictor is being asked to reconstruct.

Two deliberate simplifications vs. a fully general implementation, both
inherited from how the official facebookresearch/ijepa collator actually
works in practice:

1. Block *size* (height/width in patches) is sampled once per collator call
   and shared across the whole batch, and across all `num_target_blocks`
   target blocks in a call. Only block *position* varies per sample. This
   keeps target-block tensors a uniform (B, K) shape with no padding needed.
2. Context blocks, after target patches are removed, *do* end up with a
   variable number of kept patches per sample (since target-block overlap
   with the context region differs per sample). We handle this with
   padding + a key-padding mask (`gather_with_padding`) rather than
   dropping to a fixed size, so no sample loses more context than necessary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class MaskingConfig:
    input_size: int = 224
    patch_size: int = 16
    enc_mask_scale: tuple[float, float] = (0.85, 1.0)
    pred_mask_scale: tuple[float, float] = (0.15, 0.2)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    num_target_blocks: int = 4
    min_keep: int = 4  # minimum context patches guaranteed after target removal


class MultiBlockMaskCollator:
    def __init__(self, cfg: MaskingConfig) -> None:
        self.cfg = cfg
        self.grid_h = cfg.input_size // cfg.patch_size
        self.grid_w = cfg.input_size // cfg.patch_size
        self.num_patches = self.grid_h * self.grid_w

    def _sample_block_hw(self, scale: tuple[float, float], aspect_ratio: tuple[float, float]) -> tuple[int, int]:
        rand_scale = torch.empty(1).uniform_(*scale).item()
        area = rand_scale * self.num_patches
        log_ar = torch.empty(1).uniform_(math.log(aspect_ratio[0]), math.log(aspect_ratio[1])).item()
        ar = math.exp(log_ar)
        h = int(round(math.sqrt(area * ar)))
        w = int(round(math.sqrt(area / ar)))
        h = max(1, min(h, self.grid_h))
        w = max(1, min(w, self.grid_w))
        return h, w

    def _sample_block_mask(self, h: int, w: int) -> torch.Tensor:
        top = torch.randint(0, self.grid_h - h + 1, (1,)).item()
        left = torch.randint(0, self.grid_w - w + 1, (1,)).item()
        mask = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        mask[top : top + h, left : left + w] = True
        return mask.flatten()

    def __call__(self, images: torch.Tensor) -> dict:
        """images: (B, C, H, W), already collated/stacked.

        Returns a dict with:
            images: passthrough, unchanged
            context_mask: (B, num_patches) bool, True = kept in context
            target_masks: list of length num_target_blocks, each (B, num_patches)
                          bool, True = this patch belongs to that target block
        """
        B = images.shape[0]

        t_h, t_w = self._sample_block_hw(self.cfg.pred_mask_scale, self.cfg.aspect_ratio)
        target_masks = []
        union_target = torch.zeros(B, self.num_patches, dtype=torch.bool)
        for _ in range(self.cfg.num_target_blocks):
            batch_mask = torch.stack([self._sample_block_mask(t_h, t_w) for _ in range(B)])
            target_masks.append(batch_mask)
            union_target |= batch_mask

        e_h, e_w = self._sample_block_hw(self.cfg.enc_mask_scale, (1.0, 1.0))
        context_masks = torch.stack([self._sample_block_mask(e_h, e_w) for _ in range(B)])
        context_masks = context_masks & (~union_target)

        keep_counts = context_masks.sum(dim=1)
        under = keep_counts < self.cfg.min_keep
        if under.any():
            # Rare unlucky draw where target blocks eat most of the context
            # block -- resample those rows' context block without removing
            # targets, rather than let training see a near-empty context.
            n_bad = int(under.sum())
            context_masks[under] = torch.stack([self._sample_block_mask(e_h, e_w) for _ in range(n_bad)])

        return {
            "images": images,
            "context_mask": context_masks,
            "target_masks": target_masks,
        }


def mask_to_indices(mask: torch.Tensor) -> torch.Tensor:
    """mask: (B, N) bool with the *same* number of True entries in every row
    (true by construction for target-block masks; see module docstring).
    Returns (B, K) long patch indices.
    """
    rows = [torch.nonzero(m, as_tuple=True)[0] for m in mask]
    counts = {r.numel() for r in rows}
    if len(counts) != 1:
        raise ValueError(
            f"mask_to_indices expects a uniform True-count per row, got counts {counts}. "
            "This should only be called on target masks, never on (variable-count) context masks."
        )
    return torch.stack(rows)


def gather_with_padding(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the True-masked entries of each row of `x`, padding shorter rows.

    x: (B, N, D)
    mask: (B, N) bool, True = keep

    Returns:
        gathered: (B, K_max, D), zero-padded
        idx: (B, K_max) long, original patch index of each gathered slot
             (padding slots get index 0 -- harmless since they're masked out
             in attention via `pad_mask`)
        pad_mask: (B, K_max) bool, True = this slot is padding (ignore)

    Fully vectorized (a stable sort + torch.gather), no Python loop over the
    batch with in-place slice assignment -- that pattern (`gathered[i, :k] =
    x[i, pos]`) triggers a real MPS-backend autograd bug (`view size is not
    compatible with input tensor's size and stride`) on `loss.backward()`;
    this form only uses ops with correct MPS backward support.
    """
    B, N, D = x.shape
    counts = mask.sum(dim=1)
    k_max = int(counts.max().item())

    # Stable sort puts True (kept) positions first within each row, in their
    # original relative order -- exactly the gather order the old for-loop
    # produced, without needing to build it index-by-index.
    sort_idx = torch.argsort((~mask).long(), dim=1, stable=True)
    patch_idx = sort_idx[:, :k_max]  # (B, K_max)

    pad_mask = torch.arange(k_max, device=x.device)[None, :] >= counts[:, None]
    idx = patch_idx.masked_fill(pad_mask, 0)

    gathered = torch.gather(x, dim=1, index=patch_idx.unsqueeze(-1).expand(-1, -1, D))
    gathered = gathered.masked_fill(pad_mask.unsqueeze(-1), 0.0)

    return gathered, idx, pad_mask
