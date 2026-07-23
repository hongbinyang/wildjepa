"""Exercises the actual run_erm_baseline() training loop -- not just the
metrics functions it calls -- on the synthetic dataset. Before this, the
loop (checkpointing, resume, TensorBoard logging, per-split eval wiring) had
0% automated coverage despite being the single most-run script in this
project's history: it was only ever verified by hand, with real multi-hour
runs against the full WILDS benchmark. That verification doesn't survive a
refactor; this does, in a couple seconds, with no real data or network
needed.

`_build_resnet50` is monkeypatched to skip the real ImageNet-pretrained
weights it loads in production -- this test is about the training loop's
correctness, not torchvision's download, and staying network-free keeps it
fast and CI-safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from omegaconf import OmegaConf

from wildjepa.train import erm_baseline
from wildjepa.train.common import load_training_checkpoint


def _tiny_resnet(num_classes: int):
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _tiny_cfg(output_dir, **train_overrides):
    train = {
        "epochs": 2,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "eval_every": 1,
        "log_every": 1000,  # high enough that step-logging never fires in this tiny run
        "checkpoint_every_epochs": 1,
        "resume_from": None,
    }
    train.update(train_overrides)
    return OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "data": {
                "name": "synthetic",
                "num_species": 3,
                "images_per_species": 20,
                "image_size": 64,
                "batch_size": 4,
                "num_workers": 0,
            },
            "train": train,
        }
    )


def test_run_erm_baseline_trains_evaluates_and_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(erm_baseline, "_build_resnet50", _tiny_resnet)
    cfg = _tiny_cfg(tmp_path)

    results = erm_baseline.run_erm_baseline(cfg, torch.device("cpu"))

    # synthetic's splits are train/val/test (no id_val/id_test -- that's an
    # iwildcam-only distinction, see data/__init__.py's build_dataset)
    assert set(results) == {"val_macro_f1", "val_accuracy", "test_macro_f1", "test_accuracy"}
    for value in results.values():
        assert 0.0 <= value <= 1.0

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "erm_baseline_epoch1.pt").exists()
    assert (checkpoint_dir / "erm_baseline_epoch2.pt").exists()
    assert (checkpoint_dir / "erm_baseline_latest.pt").exists()

    # the "latest" checkpoint should reflect the final epoch, not the first
    assert load_training_checkpoint(checkpoint_dir / "erm_baseline_latest.pt")["epoch"] == 1  # 0-indexed


def test_run_erm_baseline_resume_continues_from_checkpoint_not_from_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(erm_baseline, "_build_resnet50", _tiny_resnet)

    # Train 1 epoch, then "resume" for 2 more -- if resume worked, only
    # epoch2/epoch3 checkpoints get added, epoch1 is untouched, and the loop
    # actually starts partway through rather than at epoch 1 again.
    cfg1 = _tiny_cfg(tmp_path, epochs=1)
    erm_baseline.run_erm_baseline(cfg1, torch.device("cpu"))

    checkpoint_dir = tmp_path / "checkpoints"
    epoch1_mtime = (checkpoint_dir / "erm_baseline_epoch1.pt").stat().st_mtime

    cfg2 = _tiny_cfg(
        tmp_path,
        epochs=3,
        resume_from=str(checkpoint_dir / "erm_baseline_latest.pt"),
    )
    erm_baseline.run_erm_baseline(cfg2, torch.device("cpu"))

    assert (checkpoint_dir / "erm_baseline_epoch1.pt").stat().st_mtime == epoch1_mtime, (
        "resume must not redo epoch 1 -- its checkpoint should be untouched"
    )
    assert (checkpoint_dir / "erm_baseline_epoch2.pt").exists()
    assert (checkpoint_dir / "erm_baseline_epoch3.pt").exists()


def test_auto_resume_false_ignores_existing_checkpoint_by_default(tmp_path, monkeypatch):
    """auto_resume defaults to False -- reusing a run_name (and thus an
    output_dir with an existing checkpoint) must NOT silently resume unless
    explicitly asked to. This is what keeps run_name reuse safe: an
    accidental name collision starts fresh and overwrites, rather than
    silently continuing training on unrelated old weights."""
    monkeypatch.setattr(erm_baseline, "_build_resnet50", _tiny_resnet)

    cfg1 = _tiny_cfg(tmp_path, epochs=1)
    erm_baseline.run_erm_baseline(cfg1, torch.device("cpu"))
    checkpoint_dir = tmp_path / "checkpoints"
    epoch1_mtime = (checkpoint_dir / "erm_baseline_epoch1.pt").stat().st_mtime

    # same run_name/output_dir, no resume_from, auto_resume left at its
    # False default -- should start over, not resume
    cfg2 = _tiny_cfg(tmp_path, epochs=1)
    erm_baseline.run_erm_baseline(cfg2, torch.device("cpu"))

    assert (checkpoint_dir / "erm_baseline_epoch1.pt").stat().st_mtime > epoch1_mtime, (
        "expected epoch1's checkpoint to be overwritten by a fresh run, not left alone"
    )


def test_auto_resume_true_resumes_from_latest_checkpoint_without_resume_from(tmp_path, monkeypatch):
    """With auto_resume=true and no explicit resume_from, an existing
    outputs/<run_name>/checkpoints/erm_baseline_latest.pt should be picked
    up automatically -- the whole point being you only ever need to pass
    run_name, not the checkpoint path, to continue a run."""
    monkeypatch.setattr(erm_baseline, "_build_resnet50", _tiny_resnet)

    cfg1 = _tiny_cfg(tmp_path, epochs=1)
    erm_baseline.run_erm_baseline(cfg1, torch.device("cpu"))
    checkpoint_dir = tmp_path / "checkpoints"
    epoch1_mtime = (checkpoint_dir / "erm_baseline_epoch1.pt").stat().st_mtime

    cfg2 = _tiny_cfg(tmp_path, epochs=3, auto_resume=True)
    erm_baseline.run_erm_baseline(cfg2, torch.device("cpu"))

    assert (checkpoint_dir / "erm_baseline_epoch1.pt").stat().st_mtime == epoch1_mtime, (
        "auto_resume must not redo epoch 1"
    )
    assert (checkpoint_dir / "erm_baseline_epoch2.pt").exists()
    assert (checkpoint_dir / "erm_baseline_epoch3.pt").exists()


def test_auto_resume_true_starts_fresh_when_no_checkpoint_exists(tmp_path, monkeypatch):
    """auto_resume=true on a brand-new run_name (nothing to resume from yet)
    must not error -- it should just behave like a normal fresh run."""
    monkeypatch.setattr(erm_baseline, "_build_resnet50", _tiny_resnet)
    cfg = _tiny_cfg(tmp_path, epochs=1, auto_resume=True)

    results = erm_baseline.run_erm_baseline(cfg, torch.device("cpu"))

    assert (tmp_path / "checkpoints" / "erm_baseline_epoch1.pt").exists()
    assert set(results) == {"val_macro_f1", "val_accuracy", "test_macro_f1", "test_accuracy"}
