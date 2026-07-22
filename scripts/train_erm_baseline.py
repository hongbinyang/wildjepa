#!/usr/bin/env python
"""Phase 0 sanity check: plain supervised ResNet-50 ERM on the real
iWildCam2020-WILDS benchmark, to confirm the eval harness reproduces the
published macro-F1 numbers before trusting anything built on I-JEPA.

    python scripts/train_erm_baseline.py                                  # full benchmark, published hyperparams
    python scripts/train_erm_baseline.py data=iwildcam_subset train.epochs=3   # quick smoke test
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from wildjepa.eval import print_comparison
from wildjepa.train.erm_baseline import run_erm_baseline
from wildjepa.utils.device import device_summary, resolve_device

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="erm_baseline_config")
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.device.name, cfg.device.get("allow_mps_fallback", True))
    logger.info("Using device: %s", device_summary(device))

    results = run_erm_baseline(cfg, device)
    logger.info("Results: %s", results)
    print_comparison(results)


if __name__ == "__main__":
    main()
