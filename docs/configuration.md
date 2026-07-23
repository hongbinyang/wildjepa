# Configuration reference

Every parameter in every [Hydra](https://hydra.cc) config group, organized by
category. For *commands* (which script to run for which task), see
[`usage.md`](usage.md); this document is the field-by-field reference for
*what you can set on those commands*.

All config files live under `configs/`. Any field shown here can be overridden
on the command line without editing the YAML, e.g. `train.epochs=50` or
`backend.encoder.depth=24`.

- [Top-level configs](#top-level-configs)
- [`backend`](#backend) — which JEPA implementation
- [`data`](#data) — which dataset and how it's loaded
- [`device`](#device) — compute target
- [`train`](#train) — optimization hyperparameters and run mode

---

## Top-level configs

Two entry points compose these groups differently, since the ERM baseline has
no JEPA backend:

| File | Used by | Defaults |
|---|---|---|
| `configs/config.yaml` | `scripts/train.py`, `scripts/evaluate.py` | `backend=scratch`, `device=auto`, `data=iwildcam_subset`, `train=quick_check` |
| `configs/erm_baseline_config.yaml` | `scripts/train_erm_baseline.py` | `device=auto`, `data=iwildcam_full`, `train=erm_baseline` (no `backend` group) |

Fields present in both:

| Field | Default | Meaning |
|---|---|---|
| `seed` | `42` | Global RNG seed. |
| `run_name` | `${now:%Y-%m-%d}_${now:%H-%M-%S}` | The run's identity *and* its directory name — override with `run_name=my-run` for a stable, resumable name instead of a timestamp. See [`lifecycle.md`](lifecycle.md#run-identity-what-identifies-a-run-today). |
| `output_dir` | `outputs/${run_name}` | Per-run directory, derived from `run_name`; checkpoints, Hydra logs, and TensorBoard metrics (`<output_dir>/tensorboard/`) all land here. Not usually overridden directly — set `run_name` instead. |

See [`usage.md#category-run-management`](usage.md#category-run-management) for
`scripts/manage_runs.py` (listing and deleting run directories), and
[`usage.md#category-monitoring`](usage.md#category-monitoring) for the
TensorBoard dashboard.

---

## `backend`

Which JEPA implementation to use. Selected via `backend=<name>`; see
[`design.md`](design.md) for why three backends exist.

### `backend=scratch` (default) — the from-scratch implementation

```yaml
name: scratch
encoder: {arch, img_size, patch_size, embed_dim, depth, num_heads}
predictor: {depth, embed_dim, num_heads}
ema: {momentum_start, momentum_end}
masking: {num_target_blocks, target_scale, target_aspect_ratio, context_scale}
pretrained_checkpoint: null
```

| Field | Default | Meaning |
|---|---|---|
| `encoder.arch` | `vit_small_patch16` | ViT architecture tag (informational; actual shape comes from the fields below). Small enough to iterate on quickly on an M2. |
| `encoder.img_size` | `224` | Input image resolution (pixels, square). |
| `encoder.patch_size` | `16` | ViT patch size (pixels). `img_size / patch_size` sets the patch grid used by masking. |
| `encoder.embed_dim` | `384` | Encoder transformer hidden dimension. |
| `encoder.depth` | `12` | Number of transformer blocks in the encoder. |
| `encoder.num_heads` | `6` | Attention heads per encoder block. |
| `predictor.depth` | `6` | Number of transformer blocks in the predictor. |
| `predictor.embed_dim` | `192` | Predictor hidden dimension — deliberately narrower than the encoder, per the I-JEPA paper. |
| `predictor.num_heads` | `6` | Attention heads per predictor block. |
| `ema.momentum_start` | `0.996` | Target-encoder EMA momentum at the start of training. |
| `ema.momentum_end` | `1.0` | Target-encoder EMA momentum at the end of training (momentum is scheduled to increase linearly between these). |
| `masking.num_target_blocks` | `4` | Number of target blocks sampled per image. |
| `masking.target_scale` | `[0.15, 0.2]` | Target block area, as a fraction of total patches, sampled per batch. |
| `masking.target_aspect_ratio` | `[0.75, 1.5]` | Target block aspect ratio range. |
| `masking.context_scale` | `[0.85, 1.0]` | Context block area, as a fraction of total patches. |
| `pretrained_checkpoint` | `null` | Path to a checkpoint to warm-start from — either one of this project's own pretraining checkpoints, or (with remapping) an official `facebookresearch/ijepa` `.pth`, for the cross-backend correctness check. |

### `backend=fb_ijepa` — reference: `facebookresearch/ijepa`

```yaml
name: fb_ijepa
repo_path: ${oc.env:WILDJEPA_FB_IJEPA_PATH,third_party/ijepa}
arch: vit_huge_patch14
pretrained_checkpoint: null
```

| Field | Default | Meaning |
|---|---|---|
| `repo_path` | `third_party/ijepa` (or `$WILDJEPA_FB_IJEPA_PATH` if set) | Local checkout of Meta's `facebookresearch/ijepa` repo — required, this backend imports from it directly. |
| `arch` | `vit_huge_patch14` | Architecture tag matching the reference repo's naming. |
| `pretrained_checkpoint` | `null` | Path to an official release checkpoint (e.g. `IN1K-vit.h.14-300e.pth.tar`). |

### `backend=hf_ijepa` — reference: Hugging Face `transformers`

```yaml
name: hf_ijepa
model_id: facebook/ijepa_vith14_1k
pretrained: true
pretrained_checkpoint: null
```

| Field | Default | Meaning |
|---|---|---|
| `model_id` | `facebook/ijepa_vith14_1k` | Hugging Face Hub model id, loaded via `transformers.AutoModel`. |
| `pretrained` | `true` | Whether to load pretrained weights from the Hub (vs. random init of the same architecture). |
| `pretrained_checkpoint` | `null` | Optional local checkpoint applied on top of the Hub weights above. |

For both `fb_ijepa` and `hf_ijepa`, `build_predictor` returns `None` — these
backends are used for inference/eval/correctness checks, not for running the
project's own pretraining loop (see `wildjepa/models/base.py`).

---

## `data`

Which dataset to load and how. Selected via `data=<name>`; see
[`design.md`](design.md) for the stratified-subset rationale.

### `data=synthetic` — no-download smoke-test dataset

```yaml
name: synthetic
num_species: 8
images_per_species: 64
image_size: 224
batch_size: 16
num_workers: 0
```

| Field | Default | Meaning |
|---|---|---|
| `num_species` | `8` | Number of synthetic classes (distinct colored blobs on noisy backgrounds). |
| `images_per_species` | `64` | Images generated per class. Split 70/15/15 into train/val/test. |
| `image_size` | `224` | Generated image resolution. |
| `batch_size` | `16` | DataLoader batch size. |
| `num_workers` | `0` | DataLoader worker processes (0 = load in the main process). |

### `data=iwildcam_subset` — fast-iteration real-data slice

```yaml
name: iwildcam_subset
root: data/iwildcam
num_species: 8
max_images_per_species: 500
image_size: 224
batch_size: 32
num_workers: 4
```

| Field | Default | Meaning |
|---|---|---|
| `root` | `data/iwildcam` | Where `scripts/download_data.py` put the dataset. |
| `num_species` | `8` | Restricts every split to the `num_species` most-frequent species *in the training split* (computed once, applied identically across all splits so eval labels stay meaningful). |
| `max_images_per_species` | `500` | Caps images per species per split. `null` = no cap. |
| `image_size` | `224` | Resize resolution fed to the model. |
| `batch_size` | `32` | DataLoader batch size. |
| `num_workers` | `4` | DataLoader worker processes. |

### `data=iwildcam_full` — the real benchmark

```yaml
name: iwildcam_full
root: data/iwildcam
num_species: null
max_images_per_species: null
image_size: 224
batch_size: 64
num_workers: 8
```

Same fields as `iwildcam_subset`, with `num_species`/`max_images_per_species`
set to `null` (all 182 species, no cap) — this is the split comparable to
published WILDS numbers.

Both `iwildcam_*` configs load via the real WILDS `iWildCam2020-WILDS` splits:

| Split | Images | Cameras | Role |
|---|---|---|---|
| `train` | 129,809 | 243 | Training |
| `id_val` | — | same 243 | In-distribution validation |
| `id_test` | 8,154 | same 243 | In-distribution test |
| `val` | — | 48 different | Out-of-distribution validation |
| `test` | 42,791 | 48 different | Out-of-distribution test (the headline WILDS metric) |

---

## `device`

Compute target. Selected via `device=<name>`. Resolution logic:
`wildjepa.utils.device.resolve_device` (cuda > mps > cpu when `auto`).

| `device=` | `name` field | `allow_mps_fallback` | Meaning |
|---|---|---|---|
| `auto` (default) | `auto` | `true` | Picks CUDA if available, else MPS, else CPU. |
| `cpu` | `cpu` | — | Forces CPU; raises if unavailable (never happens). |
| `cuda` | `cuda` | — | Forces CUDA; raises if `torch.cuda.is_available()` is `False`. |
| `mps` | `mps` | `true` | Forces Apple Silicon MPS; raises if unavailable. |

`allow_mps_fallback: true` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before any MPS
op runs, so operators unsupported on MPS silently fall back to CPU instead of
raising (slower, but doesn't crash the run).

---

## `train`

Run mode and optimization hyperparameters. Selected via `train=<name>`.

### `train=quick_check` (default for `scripts/train.py`)

```yaml
name: quick_check
mode: linear_probe   # linear_probe | pretrain | finetune
epochs: 5
lr: 1.0e-3
weight_decay: 0.05
eval_every: 1
log_every: 20
checkpoint_every_epochs: 1
resume_from: null
```

| Field | Default | Meaning |
|---|---|---|
| `mode` | `linear_probe` | Which training loop `scripts/train.py` dispatches to: `pretrain` (self-supervised, scratch backend only), `linear_probe` (frozen features + sklearn), or `finetune` (end-to-end). |
| `epochs` | `5` | Training epochs (ignored by `linear_probe`, which fits a single sklearn model — used by `pretrain`/`finetune`). |
| `lr` | `1.0e-3` | Optimizer learning rate (AdamW, for `pretrain`/`finetune`). |
| `weight_decay` | `0.05` | AdamW weight decay. |
| `eval_every` | `1` | Epochs between full-split evaluation passes (`finetune`). |
| `log_every` | `20` | Steps between loss log lines (`pretrain`). |
| `checkpoint_every_epochs` | `1` | `pretrain` only: save a resumable checkpoint to `<output_dir>/checkpoints/pretrain_epoch<N>.pt` (and update `pretrain_latest.pt`) every N epochs. |
| `resume_from` | `null` | `pretrain` only: path to a checkpoint to resume from — see [`usage.md`](usage.md#pausing-and-resuming-a-run). |

### `train=erm_baseline` (default for `scripts/train_erm_baseline.py`)

```yaml
name: erm_baseline
mode: erm_baseline
epochs: 12
lr: 3.0e-5
weight_decay: 0.0
eval_every: 1
log_every: 50
checkpoint_every_epochs: 1
resume_from: null
```

| Field | Default | Meaning |
|---|---|---|
| `mode` | `erm_baseline` | Fixed — this config only drives `scripts/train_erm_baseline.py`. |
| `epochs` | `12` | Matches the published WILDS paper's iWildCam ERM (ResNet-50) config, so the reproduced number is actually comparable. |
| `lr` | `3.0e-5` | Adam learning rate, matches WILDS paper. |
| `weight_decay` | `0.0` | Adam weight decay, matches WILDS paper. |
| `eval_every` | `1` | Epochs between full-split evaluation passes. |
| `log_every` | `50` | Steps between loss log lines. |
| `checkpoint_every_epochs` | `1` | Save model + optimizer + epoch index to `<output_dir>/checkpoints/erm_baseline_epoch<N>.pt` (and update `erm_baseline_latest.pt`) every N epochs. |
| `resume_from` | `null` | Path to a checkpoint to resume from. Training continues at the epoch after the one recorded in the checkpoint — see [`usage.md`](usage.md#pausing-and-resuming-a-run). |

Changing `epochs`/`lr`/`weight_decay` away from the WILDS defaults is fine for
a quick smoke test (e.g. `train.epochs=1`), but the result is then no longer a
faithful comparison point against `PUBLISHED_BASELINES` — see
[`usage.md`](usage.md#scripts-train_erm_baselinepy--phase-0-supervised-resnet-50-sanity-check).
