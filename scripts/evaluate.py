#!/usr/bin/env python
"""Load a checkpoint and run linear-probe evaluation against it, printing a
comparison to the published WILDS baseline. Separate from scripts/train.py
so re-evaluating an existing checkpoint doesn't require re-running training.

    python scripts/evaluate.py backend=scratch \\
        backend.pretrained_checkpoint=outputs/2026-.../pretrain_checkpoint.pt \\
        data=iwildcam_full
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from wildjepa.eval import print_comparison
from wildjepa.models.base import build_encoder
from wildjepa.train.linear_probe import run_linear_probe
from wildjepa.utils.device import device_summary, resolve_device

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    if not cfg.backend.get("pretrained_checkpoint"):
        raise ValueError(
            "scripts/evaluate.py requires backend.pretrained_checkpoint to be set -- "
            "pass e.g. backend.pretrained_checkpoint=outputs/.../pretrain_checkpoint.pt"
        )

    device = resolve_device(cfg.device.name, cfg.device.get("allow_mps_fallback", True))
    logger.info("Using device: %s", device_summary(device))

    encoder = build_encoder(cfg, device)
    results = run_linear_probe(cfg, device, encoder)

    logger.info("Results: %s", results)
    print_comparison(results)


if __name__ == "__main__":
    main()
