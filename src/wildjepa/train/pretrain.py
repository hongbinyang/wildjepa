"""Self-supervised I-JEPA pretraining loop. Only meaningful for
backend=scratch -- fb_ijepa/hf_ijepa are reference/inference backends, not
wired into this training loop (see docs/design.md)."""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from wildjepa.data import build_dataset, make_pretrain_collate_fn
from wildjepa.models.scratch import IJEPA
from wildjepa.models.scratch.ema import momentum_schedule
from wildjepa.models.scratch.masking import MaskingConfig
from wildjepa.train.common import AverageMeter, MetricsLogger, save_pretrain_checkpoint
from wildjepa.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _build_masking_config(cfg: DictConfig) -> MaskingConfig:
    backend = cfg.backend
    return MaskingConfig(
        input_size=backend.encoder.img_size,
        patch_size=backend.encoder.patch_size,
        enc_mask_scale=tuple(backend.masking.context_scale),
        pred_mask_scale=tuple(backend.masking.target_scale),
        aspect_ratio=tuple(backend.masking.target_aspect_ratio),
        num_target_blocks=backend.masking.num_target_blocks,
    )


def run_pretraining(cfg: DictConfig, device: torch.device) -> IJEPA:
    set_seed(cfg.seed)

    bundle = build_dataset(cfg.data)
    train_dataset = bundle.splits["train"]

    masking_cfg = _build_masking_config(cfg)
    collate_fn = make_pretrain_collate_fn(masking_cfg)
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError(
            f"Training dataset has {len(train_dataset)} examples but batch_size="
            f"{cfg.data.batch_size} with drop_last=True produces zero batches. "
            "Reduce batch_size or use a larger dataset."
        )

    model = IJEPA(cfg.backend).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    total_steps = cfg.train.epochs * len(loader)
    momentum_start = cfg.backend.ema.momentum_start
    momentum_end = cfg.backend.ema.momentum_end
    metrics_logger = MetricsLogger(cfg.output_dir)

    step = 0
    for epoch in range(cfg.train.epochs):
        meter = AverageMeter()
        for batch in loader:
            images = batch["images"].to(device)
            context_mask = batch["context_mask"].to(device)
            target_masks = [m.to(device) for m in batch["target_masks"]]

            loss = model(images, context_mask, target_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            momentum = momentum_schedule(step, total_steps, momentum_start, momentum_end)
            model.update_target_encoder(momentum)

            meter.update(loss.item(), images.size(0))
            step += 1
            metrics_logger.log_scalar("train/loss_step", loss.item(), step)
            metrics_logger.log_scalar("train/ema_momentum", momentum, step)
            if step % cfg.train.log_every == 0:
                logger.info(
                    "epoch %d/%d step %d/%d loss %.4f (avg %.4f) momentum %.5f",
                    epoch + 1,
                    cfg.train.epochs,
                    step,
                    total_steps,
                    loss.item(),
                    meter.avg,
                    momentum,
                )

        logger.info("epoch %d/%d done, avg loss %.4f", epoch + 1, cfg.train.epochs, meter.avg)
        metrics_logger.log_scalar("train/loss_epoch", meter.avg, epoch + 1)

    checkpoint_path = f"{cfg.output_dir}/pretrain_checkpoint.pt"
    save_pretrain_checkpoint(model, checkpoint_path)
    logger.info("Saved pretraining checkpoint to %s", checkpoint_path)
    metrics_logger.close()

    return model
