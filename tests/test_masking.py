import torch

from wildjepa.models.scratch.masking import (
    MaskingConfig,
    MultiBlockMaskCollator,
    gather_with_padding,
    mask_to_indices,
)


def _small_cfg(**overrides):
    defaults = dict(
        input_size=64,
        patch_size=16,  # 4x4 = 16 patch grid
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        num_target_blocks=3,
        min_keep=2,
    )
    defaults.update(overrides)
    return MaskingConfig(**defaults)


def test_collator_output_shapes():
    cfg = _small_cfg()
    collator = MultiBlockMaskCollator(cfg)
    images = torch.randn(5, 3, 64, 64)

    out = collator(images)

    assert out["images"] is images
    assert out["context_mask"].shape == (5, 16)
    assert out["context_mask"].dtype == torch.bool
    assert len(out["target_masks"]) == 3
    for tm in out["target_masks"]:
        assert tm.shape == (5, 16)
        assert tm.dtype == torch.bool


def test_target_blocks_have_uniform_count_per_sample():
    cfg = _small_cfg()
    collator = MultiBlockMaskCollator(cfg)
    images = torch.randn(8, 3, 64, 64)
    out = collator(images)

    for tm in out["target_masks"]:
        counts = tm.sum(dim=1)
        assert (counts == counts[0]).all(), "all samples should have the same target-block patch count"
        assert counts[0] > 0


def test_context_excludes_target_patches():
    cfg = _small_cfg()
    collator = MultiBlockMaskCollator(cfg)
    images = torch.randn(4, 3, 64, 64)
    out = collator(images)

    union_target = torch.zeros_like(out["context_mask"])
    for tm in out["target_masks"]:
        union_target |= tm

    overlap = out["context_mask"] & union_target
    assert not overlap.any(), "context mask must never include a target-block patch"


def test_context_respects_min_keep():
    cfg = _small_cfg(min_keep=2)
    collator = MultiBlockMaskCollator(cfg)
    images = torch.randn(16, 3, 64, 64)
    out = collator(images)

    assert (out["context_mask"].sum(dim=1) >= cfg.min_keep).all()


def test_mask_to_indices_uniform():
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[:, [1, 3, 5]] = True
    idx = mask_to_indices(mask)
    assert idx.shape == (3, 3)
    assert (idx == torch.tensor([1, 3, 5])).all()


def test_mask_to_indices_rejects_ragged():
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, [1, 2]] = True
    mask[1, [1]] = True
    try:
        mask_to_indices(mask)
        assert False, "expected ValueError for ragged mask"
    except ValueError:
        pass


def test_gather_with_padding_shapes_and_values():
    x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask = torch.tensor([[True, False, True, False, True], [True, True, False, False, False]])

    gathered, idx, pad_mask = gather_with_padding(x, mask)

    assert gathered.shape == (2, 3, 3)  # row 0 has 3 kept, row 1 has 2 (padded to 3)
    assert pad_mask.shape == (2, 3)
    assert not pad_mask[0].any()
    assert bool(pad_mask[1, 2])
    # row 0's first kept token should be x[0, 0]
    assert torch.equal(gathered[0, 0], x[0, 0])
    assert idx[0, 0].item() == 0


def test_gather_with_padding_pad_to_gives_fixed_shape_regardless_of_mask():
    """The MPS-motivated fix: passing pad_to should produce the same shape
    no matter how many patches a given mask actually keeps -- unlike the
    default (dynamic per-call) behavior tested above. See gather_with_padding's
    pad_to docs and docs/design.md "Honest limitations"."""
    x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    sparse_mask = torch.tensor([[True, False, False, False, False], [False, True, False, False, False]])
    dense_mask = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])

    sparse_out, _, sparse_pad = gather_with_padding(x, sparse_mask, pad_to=5)
    dense_out, _, dense_pad = gather_with_padding(x, dense_mask, pad_to=5)

    assert sparse_out.shape == dense_out.shape == (2, 5, 3)
    assert sparse_pad.shape == dense_pad.shape == (2, 5)
    # sanity: pad_to=5 still correctly marks the actual padding slots
    assert sparse_pad.sum(dim=1).tolist() == [4, 4]
    assert dense_pad.sum(dim=1).tolist() == [1, 0]


def test_collator_block_size_is_fixed_across_calls_not_per_call():
    """The actual bug this project hit: block size (and therefore every
    downstream tensor shape) used to be resampled on every __call__, which
    meant a new shape on nearly every training batch -- catastrophic on
    MPS, which recompiles its graph per shape. Block size must now be fixed
    once, at construction, and stay identical across every call for the
    life of the collator."""
    cfg = _small_cfg()
    collator = MultiBlockMaskCollator(cfg)
    images = torch.randn(4, 3, 64, 64)

    first_counts = [tm.sum(dim=1)[0].item() for tm in collator(images)["target_masks"]]
    for _ in range(10):
        counts = [tm.sum(dim=1)[0].item() for tm in collator(images)["target_masks"]]
        assert counts == first_counts, "target-block size must stay identical across calls"
