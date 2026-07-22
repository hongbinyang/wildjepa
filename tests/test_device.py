import pytest
import torch

from wildjepa.utils.device import device_summary, resolve_device


def test_resolve_auto_returns_available_device():
    device = resolve_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}


def test_resolve_cpu_always_works():
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_resolve_unavailable_device_raises():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine; can't test the unavailable path")
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_device_summary_is_a_string():
    device = resolve_device("cpu")
    assert isinstance(device_summary(device), str)
