import torch
import torch.nn as nn

from wildjepa.train.common import (
    AverageMeter,
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_average_meter_tracks_running_average():
    meter = AverageMeter()
    assert meter.avg == 0.0  # no updates yet -- must not divide by zero

    meter.update(2.0, n=1)
    meter.update(4.0, n=1)
    assert meter.avg == 3.0


def test_average_meter_weights_by_n():
    meter = AverageMeter()
    meter.update(1.0, n=3)  # e.g. a batch-size-weighted loss
    meter.update(5.0, n=1)
    assert meter.avg == 2.0  # (1*3 + 5*1) / 4


def test_average_meter_reset_clears_state():
    meter = AverageMeter()
    meter.update(10.0, n=5)
    meter.reset()
    assert meter.avg == 0.0


def test_save_and_load_training_checkpoint_round_trips(tmp_path):
    """save_training_checkpoint/load_training_checkpoint back the erm_baseline
    and pretrain resume features -- both were verified by hand with real
    training runs, but that verification doesn't survive a refactor. This is
    the fast automated equivalent: build a model+optimizer, save, load into
    fresh instances, confirm identical state."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Take one optimizer step so its state dict isn't just empty defaults.
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    optimizer.step()

    path = tmp_path / "checkpoints" / "run_epoch2.pt"
    save_training_checkpoint(path, epoch=2, model=model, optimizer=optimizer)
    assert path.exists()

    state = load_training_checkpoint(path)
    assert state["epoch"] == 2

    fresh_model = nn.Linear(4, 2)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    fresh_model.load_state_dict(state["model"])
    fresh_optimizer.load_state_dict(state["optimizer"])

    for p1, p2 in zip(model.parameters(), fresh_model.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert fresh_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()


def test_save_training_checkpoint_creates_parent_directory(tmp_path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "nested" / "does" / "not" / "exist" / "ckpt.pt"

    save_training_checkpoint(path, epoch=0, model=model, optimizer=optimizer)

    assert path.exists()
