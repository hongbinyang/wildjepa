import torch

from wildjepa.models.scratch.loss import jepa_loss


def test_zero_loss_for_identical_predictions():
    t = [torch.randn(2, 4, 8) for _ in range(3)]
    loss = jepa_loss(t, [x.clone() for x in t])
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_loss_is_positive_for_different_tensors():
    preds = [torch.zeros(2, 4, 8)]
    targets = [torch.ones(2, 4, 8)]
    loss = jepa_loss(preds, targets)
    assert loss.item() > 0


def test_mismatched_lengths_raise():
    try:
        jepa_loss([torch.zeros(1, 1, 1)], [])
        assert False, "expected ValueError"
    except ValueError:
        pass
