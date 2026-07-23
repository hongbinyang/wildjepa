import pytest
import torch

from wildjepa.models.scratch.patch_embed import PatchEmbed
from wildjepa.models.scratch.pos_embed import get_2d_sincos_pos_embed
from wildjepa.models.scratch.vit import VisionTransformer
from wildjepa.utils.device import resolve_device


def test_pos_embed_shape():
    pe = get_2d_sincos_pos_embed(embed_dim=32, grid_size=4)
    assert pe.shape == (16, 32)


def test_pos_embed_rejects_non_divisible_dim():
    with pytest.raises(ValueError):
        get_2d_sincos_pos_embed(embed_dim=30, grid_size=4)


def test_patch_embed_shape():
    pe = PatchEmbed(img_size=64, patch_size=16, in_chans=3, embed_dim=32)
    x = torch.randn(2, 3, 64, 64)
    out = pe(x)
    assert out.shape == (2, 16, 32)  # (64/16)^2 = 16 patches


def test_patch_embed_rejects_wrong_size():
    pe = PatchEmbed(img_size=64, patch_size=16, in_chans=3, embed_dim=32)
    with pytest.raises(ValueError):
        pe(torch.randn(2, 3, 32, 32))


def test_patch_embed_backward_on_resolved_device():
    """Every other test in this file only checks forward shapes on CPU --
    none call .backward(), and none use a non-CPU device. That combination
    is exactly how a real PyTorch/MPS bug (Conv2d's backward breaking
    whenever its output feeds a further op after a transpose -- precisely
    what PatchEmbed.forward's callers always do next) went undetected until
    pretraining was run on real Apple Silicon hardware for the first time --
    see docs/design.md "Honest limitations". Uses resolve_device("auto")
    rather than hardcoding mps/cuda, so this is safe to run anywhere: a
    GPU-less CI runner just re-exercises cpu here, harmlessly."""
    device = resolve_device("auto")
    pe = PatchEmbed(img_size=64, patch_size=16, in_chans=3, embed_dim=32).to(device)
    x = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)

    out = pe(x)
    out.sum().backward()  # raised RuntimeError on MPS before the fix

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def _tiny_vit():
    return VisionTransformer(img_size=64, patch_size=16, embed_dim=32, depth=2, num_heads=4)


def test_forward_full_shape():
    vit = _tiny_vit()
    x = torch.randn(3, 3, 64, 64)
    out = vit.forward_full(x)
    assert out.shape == (3, 16, 32)


def test_forward_masked_shape_and_padding():
    vit = _tiny_vit()
    x = torch.randn(3, 3, 64, 64)
    mask = torch.zeros(3, 16, dtype=torch.bool)
    mask[0, :10] = True
    mask[1, :5] = True
    mask[2, :10] = True

    tokens, idx, pad_mask = vit.forward_masked(x, mask)

    assert tokens.shape == (3, 10, 32)  # padded to the max (10)
    assert idx.shape == (3, 10)
    assert pad_mask.shape == (3, 10)
    assert pad_mask[1, 5:].all()  # sample 1 only had 5 real tokens
    assert not pad_mask[0].any()
    assert not pad_mask[2].any()


def test_forward_full_is_deterministic_in_eval_mode():
    vit = _tiny_vit()
    vit.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out1 = vit.forward_full(x)
        out2 = vit.forward_full(x)
    assert torch.equal(out1, out2)


def test_forward_masked_backward_on_resolved_device():
    """Same rationale as test_patch_embed_backward_on_resolved_device --
    exercises the full masking -> attention chain's backward on this
    machine's actual resolved device, not just patch embedding alone."""
    device = resolve_device("auto")
    vit = _tiny_vit().to(device)
    x = torch.randn(3, 3, 64, 64, device=device, requires_grad=True)
    mask = torch.zeros(3, 16, dtype=torch.bool, device=device)
    mask[0, :10] = True
    mask[1, :5] = True
    mask[2, :10] = True

    tokens, _idx, _pad_mask = vit.forward_masked(x, mask)
    tokens.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
