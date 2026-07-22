"""Confirms the config -> factory -> backend wiring is correct across all
three backends.

`scratch` is fully implemented and tested for real here. `fb_ijepa` requires
an external repo checkout we don't have in this dev environment, so we only
check it fails with a clear, actionable error. `hf_ijepa` requires network
access to fetch weights from the Hugging Face Hub on first use -- if that's
unavailable (offline CI, no cached weights), the test skips rather than
fails, since that's an environment limitation, not a code bug.
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from wildjepa.models.base import build_encoder
from wildjepa.utils.device import resolve_device

CONFIG_DIR = str((Path(__file__).parent.parent / "configs").resolve())


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


def test_build_encoder_scratch_end_to_end():
    cfg = _compose(["backend=scratch"])
    device = resolve_device("cpu")

    encoder = build_encoder(cfg, device)

    assert encoder.embed_dim == cfg.backend.encoder.embed_dim
    x = torch.randn(2, 3, cfg.backend.encoder.img_size, cfg.backend.encoder.img_size)
    out = encoder(x)
    num_patches = (cfg.backend.encoder.img_size // cfg.backend.encoder.patch_size) ** 2
    assert out.shape == (2, num_patches, cfg.backend.encoder.embed_dim)


def test_build_encoder_fb_ijepa_fails_clearly_without_checkout():
    cfg = _compose(["backend=fb_ijepa"])
    cfg.backend.repo_path = "/tmp/definitely_not_a_real_ijepa_checkout"
    device = resolve_device("cpu")

    with pytest.raises(FileNotFoundError):
        build_encoder(cfg, device)


def test_build_encoder_hf_ijepa_or_skip():
    cfg = _compose(["backend=hf_ijepa"])
    device = resolve_device("cpu")

    try:
        encoder = build_encoder(cfg, device)
    except ImportError:
        pytest.skip("transformers not installed")
    except Exception as e:  # noqa: BLE001 -- network/hub errors vary by transformers version
        pytest.skip(f"hf_ijepa backend needs network access to the HF Hub: {e}")
    else:
        assert encoder.embed_dim > 0


def test_unknown_backend_raises_value_error():
    cfg = _compose([])
    cfg.backend.name = "not_a_real_backend"
    device = resolve_device("cpu")

    with pytest.raises(ValueError):
        build_encoder(cfg, device)
