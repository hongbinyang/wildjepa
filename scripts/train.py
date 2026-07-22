#!/usr/bin/env python
"""Entry point. Mode comes from `cfg.train.mode` (linear_probe | pretrain | finetune):

    python scripts/train.py                                        # pretrain scratch backend, subset data
    python scripts/train.py train=quick_check train.mode=pretrain
    python scripts/train.py train.mode=linear_probe backend=hf_ijepa
    python scripts/train.py train.mode=finetune backend=scratch \\
        backend.pretrained_checkpoint=outputs/.../pretrain_checkpoint.pt
    python scripts/train.py backend=fb_ijepa device=cuda data=iwildcam_full
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from wildjepa.eval import print_comparison
from wildjepa.utils.device import device_summary, resolve_device

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.device.name, cfg.device.get("allow_mps_fallback", True))
    logger.info("Using device: %s", device_summary(device))

    mode = cfg.train.mode
    if mode == "pretrain":
        from wildjepa.train.pretrain import run_pretraining

        run_pretraining(cfg, device)

    elif mode in ("linear_probe", "finetune"):
        from wildjepa.models.base import build_encoder

        encoder = build_encoder(cfg, device)

        if mode == "linear_probe":
            from wildjepa.train.linear_probe import run_linear_probe

            results = run_linear_probe(cfg, device, encoder)
        else:
            from wildjepa.train.finetune import run_finetune

            results = run_finetune(cfg, device, encoder)

        logger.info("Results: %s", results)
        print_comparison(results)

    else:
        raise ValueError(f"Unknown train.mode: {mode!r} (expected pretrain | linear_probe | finetune)")


if __name__ == "__main__":
    main()
