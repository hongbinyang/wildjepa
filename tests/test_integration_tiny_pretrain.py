"""End-to-end integration test: a tiny IJEPA trained for a few dozen steps on
the synthetic dataset should produce a finite, decreasing loss -- without the
real (large) iWildCam download. Exercises masking -> context encoding ->
target encoding -> prediction -> loss -> backward -> EMA update as one path,
on data small enough to genuinely overfit in seconds. Run twice: once
explicitly on CPU, once on whatever device this machine's own
resolve_device("auto") picks (mps on Apple Silicon, cuda if available, cpu
otherwise) -- the device-aware variant is what should have caught (and now
would catch) a real PyTorch/MPS backward bug that slipped past this suite
until pretraining was run manually on real hardware for the first time; see
docs/design.md "Honest limitations".

This does NOT prove the full-scale pipeline gets good iWildCam macro-F1 --
only that the training loop is mechanically correct. See docs/roadmap.md for
what's still needed before real benchmark numbers.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from wildjepa.data.collate import make_pretrain_collate_fn
from wildjepa.data.synthetic import SyntheticCameraTrapDataset
from wildjepa.models.scratch import IJEPA
from wildjepa.models.scratch.ema import momentum_schedule
from wildjepa.models.scratch.masking import MaskingConfig
from wildjepa.utils.device import resolve_device


def _tiny_backend_cfg():
    return OmegaConf.create(
        {
            "name": "scratch",
            "encoder": {
                "arch": "vit_tiny",
                "img_size": 64,
                "patch_size": 16,
                "embed_dim": 32,
                "depth": 2,
                "num_heads": 4,
            },
            "predictor": {"depth": 2, "embed_dim": 16, "num_heads": 4},
            "ema": {"momentum_start": 0.99, "momentum_end": 0.999},
            "masking": {
                "num_target_blocks": 2,
                "target_scale": [0.2, 0.3],
                "target_aspect_ratio": [0.75, 1.5],
                "context_scale": [0.85, 1.0],
            },
            "pretrained_checkpoint": None,
        }
    )


def _train_tiny_ijepa(device: torch.device, total_steps: int = 40) -> list[float]:
    torch.manual_seed(0)
    cfg = _tiny_backend_cfg()

    dataset = SyntheticCameraTrapDataset(num_classes=4, num_images_per_class=4, image_size=64)
    masking_cfg = MaskingConfig(
        input_size=cfg.encoder.img_size,
        patch_size=cfg.encoder.patch_size,
        enc_mask_scale=tuple(cfg.masking.context_scale),
        pred_mask_scale=tuple(cfg.masking.target_scale),
        aspect_ratio=tuple(cfg.masking.target_aspect_ratio),
        num_target_blocks=cfg.masking.num_target_blocks,
    )
    collate_fn = make_pretrain_collate_fn(masking_cfg)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, drop_last=True)

    model = IJEPA(cfg).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=5e-3)

    losses: list[float] = []
    step = 0
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break

            images = batch["images"].to(device)
            context_mask = batch["context_mask"].to(device)
            target_masks = [m.to(device) for m in batch["target_masks"]]

            loss = model(images, context_mask, target_masks)
            assert torch.isfinite(loss), f"loss is not finite at step {step}: {loss.item()}"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            momentum = momentum_schedule(step, total_steps, cfg.ema.momentum_start, cfg.ema.momentum_end)
            model.update_target_encoder(momentum)

            losses.append(loss.item())
            step += 1

    return losses


def _assert_loss_decreased(losses: list[float], total_steps: int) -> None:
    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    assert late_avg < early_avg, (
        f"expected loss to decrease over {total_steps} steps on this tiny, easily-overfit "
        f"dataset: early={early_avg:.4f} late={late_avg:.4f} (full trace: {losses})"
    )


def test_tiny_pretrain_loss_decreases_and_stays_finite():
    losses = _train_tiny_ijepa(torch.device("cpu"))
    _assert_loss_decreased(losses, total_steps=40)


def test_tiny_pretrain_loss_decreases_on_resolved_device():
    """Same as the CPU test above, but on whatever device
    wildjepa.utils.device.resolve_device() actually picks for this machine
    (mps on Apple Silicon, cuda if available, cpu otherwise). This is the
    test that should have caught a real PyTorch/MPS bug in PatchEmbed's
    backward -- every test in this suite ran on CPU by default, so
    pretraining had never actually been exercised on MPS until it was run
    manually and immediately crashed (see docs/design.md "Honest
    limitations"). Uses resolve_device("auto") rather than hardcoding a
    backend, so this is safe in any environment: a GPU-less CI runner just
    re-exercises cpu here, harmlessly."""
    device = resolve_device("auto")
    losses = _train_tiny_ijepa(device)
    _assert_loss_decreased(losses, total_steps=40)


def test_checkpoint_round_trip_produces_matching_encoder():
    """save_pretrain_checkpoint -> ScratchEncoder.load_pretrained should
    reproduce the pretrained context encoder's outputs exactly."""
    import tempfile
    from pathlib import Path

    from wildjepa.models.scratch import ScratchEncoder
    from wildjepa.train.common import save_pretrain_checkpoint

    torch.manual_seed(0)
    cfg = _tiny_backend_cfg()
    model = IJEPA(cfg)
    model.eval()

    x = torch.randn(2, 3, cfg.encoder.img_size, cfg.encoder.img_size)
    with torch.no_grad():
        expected = model.context_encoder.forward_full(x)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "checkpoint.pt"
        save_pretrain_checkpoint(model, ckpt_path)

        encoder = ScratchEncoder(cfg)
        encoder.load_pretrained(str(ckpt_path))
        encoder.eval()

        with torch.no_grad():
            actual = encoder(x)

    assert torch.allclose(expected, actual, atol=1e-6)
