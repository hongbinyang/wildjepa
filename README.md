# WildJEPA

Self-supervised, JEPA-based species classification for camera-trap wildlife imagery.

## Why

Camera traps generate huge volumes of unlabeled images, but labeled examples for rare/long-tail
species are scarce, and models trained on one set of cameras generalize poorly to new camera
deployments (background, lighting, angle overfitting). I-JEPA-style masked latent prediction is
a plausible fix: it learns semantic representations without pixel-level reconstruction or
contrastive-pair heuristics, so pretraining on abundant *unlabeled* camera-trap footage should
transfer better to new locations and rare classes than supervised training alone.

This project is a from-scratch implementation of I-JEPA (encoder, predictor, EMA target
encoder, multi-block masking), built so it can be swapped against reference implementations
(Meta's `facebookresearch/ijepa`, Hugging Face `transformers`) for correctness validation, and
evaluated against the standardized WILDS `iWildCam2020-WILDS` benchmark.

See `docs/design.md` for architecture rationale and `docs/roadmap.md` for what's next after the
initial feasibility check.

## Status

Quick-feasibility-check phase, fully implemented: ViT + masking + predictor + EMA + loss
(from scratch), real WILDS `iWildCam2020-WILDS` data loading + a no-download synthetic dataset
for smoke tests, pretrain/linear-probe/fine-tune loops, eval metrics, and real `fb_ijepa`/
`hf_ijepa` adapters. `pytest tests/` passes (42/42) on real Apple Silicon hardware. The real
`iWildCam2020-WILDS` data is downloaded, and a Phase 0 supervised ResNet-50 ERM baseline
(`scripts/train_erm_baseline.py`) is in place to validate the eval harness against the
published benchmark numbers before trusting any I-JEPA result built on top of it.

Not yet done: cross-backend checkpoint diff, and first real I-JEPA linear-probe numbers.
See `docs/roadmap.md` for the exact remaining steps.

## Setup

```bash
conda env create -f environment.yml
conda activate wildjepa
pip install -e .
pytest tests/
```

Device is auto-detected at runtime (CUDA > MPS > CPU). On Apple Silicon, set
`PYTORCH_ENABLE_MPS_FALLBACK=1` before running — some ops still fall back to CPU on MPS.

## Documentation

| Doc | Covers |
|---|---|
| [`docs/design.md`](docs/design.md) | Why I-JEPA, why from scratch, architecture rationale, evaluation protocol. |
| [`docs/roadmap.md`](docs/roadmap.md) | Phased plan and current status of each item. |
| [`docs/usage.md`](docs/usage.md) | Every command, organized by category (setup, data, training, evaluation, testing). |
| [`docs/configuration.md`](docs/configuration.md) | Every Hydra config parameter, organized by group (`backend` / `data` / `device` / `train`). |
| [`docs/lifecycle.md`](docs/lifecycle.md) | The same material organized by time instead: setup → smoke test → data → baseline → pretrain → verify → evaluate → iterate, plus open run-management design questions. |

## Project layout

```
configs/          Hydra configs (backend / device / data / train, composed via defaults list)
src/wildjepa/
  models/          JEPAEncoder / JEPAPredictor interfaces + three backends:
    base.py           scratch (our implementation), fb_ijepa, hf_ijepa
  data/            iWildCam / WILDS dataset wrappers, multi-block masking collator
  train/           pretrain / linear-probe / fine-tune loops
  eval/            macro-F1, per-class, few-shot metrics; WILDS baseline comparison
  utils/           device detection, logging
notebooks/         exploration
scripts/           CLI entry points (train.py, train_erm_baseline.py, evaluate.py, download_data.py)
tests/             unit tests, incl. cross-backend output diffing for correctness checks
docs/              design rationale, roadmap, usage + configuration reference
```

## License

Code in this repository is MIT licensed (see `LICENSE`). Pretrained checkpoints from
`facebookresearch/ijepa` (Meta) carry their own non-commercial license terms — see that repo
before using its weights beyond research/evaluation purposes.
