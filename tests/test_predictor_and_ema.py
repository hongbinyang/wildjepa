import copy

import torch

from wildjepa.models.scratch.ema import momentum_schedule, update_ema
from wildjepa.models.scratch.predictor import Predictor
from wildjepa.models.scratch.vit import VisionTransformer


def test_predictor_output_shapes():
    predictor = Predictor(
        encoder_embed_dim=32,
        predictor_embed_dim=16,
        depth=2,
        num_heads=4,
        num_patches=16,  # 4x4 grid
    )
    B, Kc = 3, 7
    context_tokens = torch.randn(B, Kc, 32)
    context_idx = torch.randint(0, 16, (B, Kc))
    context_pad_mask = torch.zeros(B, Kc, dtype=torch.bool)

    target_idx_list = [torch.randint(0, 16, (B, 4)) for _ in range(2)]

    preds = predictor(context_tokens, context_idx, context_pad_mask, target_idx_list)

    assert len(preds) == 2
    for p in preds:
        assert p.shape == (B, 4, 32)  # back in encoder_embed_dim


def test_predictor_ignores_padded_context_tokens():
    """Two calls that differ only in the *values* at padded context
    positions (with pad_mask correctly marking them) should produce the same
    prediction -- padding must not leak into attention."""
    torch.manual_seed(0)
    predictor = Predictor(encoder_embed_dim=16, predictor_embed_dim=16, depth=2, num_heads=4, num_patches=16)
    predictor.eval()

    B, Kc = 1, 4
    context_tokens_a = torch.randn(B, Kc, 16)
    context_idx = torch.tensor([[0, 1, 2, 3]])
    pad_mask = torch.tensor([[False, False, True, True]])  # last two are padding
    target_idx_list = [torch.tensor([[5, 6]])]

    context_tokens_b = context_tokens_a.clone()
    context_tokens_b[:, 2:] = torch.randn(B, 2, 16)  # change only the padded slots

    with torch.no_grad():
        pred_a = predictor(context_tokens_a, context_idx, pad_mask, target_idx_list)[0]
        pred_b = predictor(context_tokens_b, context_idx, pad_mask, target_idx_list)[0]

    assert torch.allclose(pred_a, pred_b, atol=1e-5)


def test_update_ema_moves_target_toward_context():
    vit_kwargs = dict(img_size=64, patch_size=16, embed_dim=16, depth=1, num_heads=2)
    context = VisionTransformer(**vit_kwargs)
    target = copy.deepcopy(context)

    # Perturb context so it differs from target.
    with torch.no_grad():
        for p in context.parameters():
            p.add_(torch.randn_like(p))

    before = [p.clone() for p in target.parameters()]
    update_ema(context, target, momentum=0.9)
    after = list(target.parameters())

    moved = any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
    assert moved, "target encoder parameters should change after an EMA update"

    # With momentum=0.9, target should have moved 10% of the way toward context.
    for p_ctx, p_before, p_after in zip(context.parameters(), before, after, strict=True):
        expected = 0.9 * p_before + 0.1 * p_ctx
        assert torch.allclose(p_after, expected, atol=1e-6)


def test_update_ema_momentum_one_is_a_no_op():
    vit_kwargs = dict(img_size=64, patch_size=16, embed_dim=16, depth=1, num_heads=2)
    context = VisionTransformer(**vit_kwargs)
    target = copy.deepcopy(context)
    with torch.no_grad():
        for p in context.parameters():
            p.add_(torch.randn_like(p))

    before = [p.clone() for p in target.parameters()]
    update_ema(context, target, momentum=1.0)
    after = list(target.parameters())

    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a)


def test_momentum_schedule_bounds_and_linearity():
    assert momentum_schedule(0, 100, 0.9, 1.0) == 0.9
    assert momentum_schedule(100, 100, 0.9, 1.0) == 1.0
    assert abs(momentum_schedule(50, 100, 0.9, 1.0) - 0.95) < 1e-9
    # clamped beyond total_steps
    assert momentum_schedule(1000, 100, 0.9, 1.0) == 1.0


def test_momentum_schedule_zero_total_steps_returns_end():
    assert momentum_schedule(0, 0, 0.9, 1.0) == 1.0
