import pytest
import torch

from wildjepa.models.scratch.patch_embed import PatchEmbed
from wildjepa.models.scratch.pos_embed import get_2d_sincos_pos_embed
from wildjepa.models.scratch.vit import VisionTransformer


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
