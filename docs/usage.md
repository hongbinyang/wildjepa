# Usage

Command reference, organized by what you're trying to do. For what every individual
parameter means and its default, see [`configuration.md`](configuration.md) — this
document focuses on *commands*; that one focuses on *parameters*.

All commands assume:

```bash
cd /path/to/wildjepa
conda env create -f environment.yml   # first time only
conda activate wildjepa
pip install -e .
```

Every script here is a [Hydra](https://hydra.cc) entry point. Config groups
(`backend`, `device`, `data`, `train`) compose independently and can be overridden
on the command line with `group=option` or `group.field=value` — see
[Composing configs](#composing-configs-backend--device--data--train) at the bottom.

---

## Category: Setup & Testing

### Install and verify

```bash
conda env create -f environment.yml
conda activate wildjepa
pip install -e .
pytest tests/
```

| Command | Purpose |
|---|---|
| `conda env create -f environment.yml` | Creates the `wildjepa` conda env (PyTorch 2.3+, picks CUDA/MPS/CPU build automatically per host). |
| `pip install -e .` | Installs this package in editable mode so `import wildjepa` resolves to `src/wildjepa/`. |
| `pytest tests/` | Runs the full test suite (42 tests). No external dependencies beyond `environment.yml`, except: `test_backend_factory.py::test_build_encoder_hf_ijepa_or_skip` needs network access to the Hugging Face Hub (skips otherwise). |

Device is auto-detected at runtime (CUDA > MPS > CPU). On Apple Silicon,
`PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically by
`wildjepa.utils.device.resolve_device` — some ops still fall back to CPU on MPS.

---

## Category: Data

### `scripts/download_data.py` — one-time real-data download

```bash
python scripts/download_data.py --root data/iwildcam
```

| Parameter | Default | Meaning |
|---|---|---|
| `--root` | `data/iwildcam` | Directory the `iWildCam2020-WILDS` package downloads and extracts into. |

Multi-GB download from the WILDS/CodaLab archive — deliberately a separate,
explicit step, not something that runs as a side effect of the first training
command. Verify disk space before running (dataset + image files land at ~12GB).

### Smoke-test the pipeline (no download, seconds)

Before touching real data, confirm the whole loop — masking, context/target
encoding, prediction, loss, EMA update — actually runs and learns something,
using the self-contained synthetic dataset (`wildjepa.data.synthetic`, no
internet required):

```bash
python scripts/train.py data=synthetic train.mode=pretrain train.epochs=5
```

Then linear-probe the checkpoint it produced:

```bash
python scripts/train.py data=synthetic train.mode=linear_probe \
    backend.pretrained_checkpoint=outputs/<run_name>/pretrain_checkpoint.pt
```

You should see macro-F1 well above chance (1/num_species) — if not, something in
the pipeline is broken, not just under-trained. `tests/test_integration_tiny_pretrain.py`
is the automated equivalent of this check.

### Choosing a dataset (`data=...`)

| `data=` value | Species | Images | Use for |
|---|---|---|---|
| `synthetic` | 8 (configurable) | 64/species, generated | No-download pipeline smoke test |
| `iwildcam_subset` | 8 most-frequent | ≤500/species | Fast iteration during development |
| `iwildcam_full` | all 182 | full benchmark | Real benchmark numbers, comparable to published baselines |

Full field-level reference for each: [`configuration.md#data`](configuration.md#data).

---

## Category: Training

All training goes through `scripts/train.py`; the mode is set by `train.mode`
(`pretrain` | `linear_probe` | `finetune`), or use the dedicated
`scripts/train_erm_baseline.py` for the supervised baseline (see below).

### `scripts/train.py train.mode=pretrain` — self-supervised I-JEPA pretraining

Scratch backend only (`fb_ijepa`/`hf_ijepa` are reference/inference backends, not
trained here — see [`design.md`](design.md)).

```bash
# subset: 8 species, capped images per species -- for fast iteration
python scripts/train.py data=iwildcam_subset train.mode=pretrain \
    train.epochs=20 device=mps

# full benchmark: all 182 species, no cap
python scripts/train.py data=iwildcam_full train.mode=pretrain \
    train.epochs=100 device=cuda data.batch_size=64
```

Checkpoints land at `outputs/<run_name>/pretrain_checkpoint.pt` (Hydra's
per-run output directory; `run_name` defaults to a timestamp, or pass
`run_name=my-run` for a stable name — see
[Category: Run management](#category-run-management) below).

Key overridable parameters (full list: [`configuration.md#train`](configuration.md#train)
and [`configuration.md#backend`](configuration.md#backend)):

| Parameter | Meaning |
|---|---|
| `train.epochs` | Number of pretraining epochs. |
| `train.lr` | Optimizer learning rate (AdamW). |
| `train.weight_decay` | AdamW weight decay. |
| `train.log_every` | Steps between loss log lines. |
| `backend.encoder.*` | ViT architecture (depth, embed_dim, num_heads, patch_size). |
| `backend.masking.*` | Multi-block masking geometry (target/context scale, aspect ratio, num target blocks). |
| `backend.ema.*` | Target-encoder EMA momentum schedule. |

### `scripts/train.py train.mode=linear_probe` — frozen-encoder evaluation

Standard SSL evaluation protocol: freeze the encoder, extract features, fit an
sklearn `LogisticRegression` head. Works with any backend.

```bash
python scripts/train.py train.mode=linear_probe backend=scratch \
    backend.pretrained_checkpoint=outputs/.../pretrain_checkpoint.pt \
    data=iwildcam_full
```

### `scripts/train.py train.mode=finetune` — end-to-end fine-tuning

Unfreezes the encoder, trains an `EncoderWithHead` (encoder + linear head)
jointly with cross-entropy. The upper-bound evaluation, after linear-probe.

```bash
python scripts/train.py data=iwildcam_full train.mode=finetune \
    backend.pretrained_checkpoint=outputs/.../pretrain_checkpoint.pt \
    train.epochs=10 train.lr=1e-4
```

| Parameter | Meaning |
|---|---|
| `train.epochs` | Fine-tuning epochs. |
| `train.lr` | AdamW learning rate for the whole (unfrozen) model. |
| `train.weight_decay` | AdamW weight decay. |
| `train.eval_every` | Epochs between full-split evaluation passes. |
| `backend.pretrained_checkpoint` | Required — path to a pretraining checkpoint to start from. |

### `scripts/train_erm_baseline.py` — Phase 0 supervised ResNet-50 sanity check

Plain supervised ResNet-50 (ImageNet-pretrained backbone, cross-entropy),
independent of the JEPA backend abstraction. Establishes that the eval harness
(macro-F1, split handling) reproduces the *published* WILDS ERM numbers before
trusting any I-JEPA result built on top of it — see
[`design.md`](design.md#evaluation-protocol) and
[`roadmap.md`](roadmap.md#phase-0--scaffold).

```bash
# full benchmark, published WILDS hyperparameters (default)
python scripts/train_erm_baseline.py

# quick smoke test on the subset
python scripts/train_erm_baseline.py data=iwildcam_subset train.epochs=3
```

| Parameter | Default | Meaning |
|---|---|---|
| `data` | `iwildcam_full` | Which dataset config to train on (this script's default differs from `scripts/train.py`'s, since a baseline check is only meaningful on the full 182-species benchmark). |
| `train.epochs` | `12` | Matches the WILDS paper's iWildCam ERM config. |
| `train.lr` | `3.0e-5` | Adam learning rate, matches WILDS paper. |
| `train.weight_decay` | `0.0` | Adam weight decay, matches WILDS paper. |
| `train.eval_every` | `1` | Epochs between full-split evaluation passes. |
| `train.log_every` | `50` | Steps between loss log lines. |
| `train.checkpoint_every_epochs` | `1` | Save a resumable checkpoint after every N epochs. |
| `train.resume_from` | `null` | Path to a `checkpoints/erm_baseline_*.pt` to resume from. |

This script uses its own top-level Hydra config
(`configs/erm_baseline_config.yaml`) since it has no `backend` group — see
[`configuration.md#top-level-configs`](configuration.md#top-level-configs).

#### Pausing and resuming a run

Every `checkpoint_every_epochs` epochs, model + optimizer + epoch index are saved to
`<output_dir>/checkpoints/erm_baseline_epoch<N>.pt`, and `erm_baseline_latest.pt` is
updated to always point at the most recent one. `<output_dir>` is
`outputs/<run_name>` — give the run an explicit name so the checkpoint path is
predictable and stable across resumes (see
[Category: Run management](#category-run-management)).

- **Temporary pause** (keep it in memory, resume in the same session): send the
  running process `SIGSTOP`, then `SIGCONT` when ready to continue — no code
  support needed, this is a plain OS-level freeze.
- **Full stop and resume later** (process killed, machine rebooted, etc.): re-run
  with the same `run_name` and `train.resume_from` pointing at the last
  checkpoint. Training picks up at the epoch *after* the one recorded in the
  checkpoint — it does not re-run completed epochs.

```bash
# named run, killed partway through (Ctrl-C or `kill`)
python scripts/train_erm_baseline.py run_name=erm-baseline-v1

# resume it -- same run_name keeps writing into the same outputs/erm-baseline-v1/
python scripts/train_erm_baseline.py run_name=erm-baseline-v1 \
    train.resume_from=outputs/erm-baseline-v1/checkpoints/erm_baseline_latest.pt
```

Only `scripts/train_erm_baseline.py` has checkpoint/resume today —
`scripts/train.py`'s `pretrain`/`finetune` modes save a final checkpoint but
don't yet support resuming a killed run (see `docs/roadmap.md`).

---

## Category: Run management

### Naming a run

Every training command's `run_name` doubles as its output directory name
(`outputs/<run_name>/`) — it defaults to a timestamp, but naming it explicitly
gives you a stable, predictable path for checkpoints, resuming, and cleanup:

```bash
python scripts/train_erm_baseline.py run_name=erm-baseline-v1
# -> outputs/erm-baseline-v1/checkpoints/, outputs/erm-baseline-v1/pretrain_checkpoint.pt, etc.
```

There's no separate registry beyond the filesystem — `outputs/` *is* the list
of runs (see [`lifecycle.md`](lifecycle.md#run-identity-what-identifies-a-run-today)
for why that's a deliberate choice at this project phase, not an oversight).

### `scripts/manage_runs.py` — list and delete run directories

```bash
python scripts/manage_runs.py list
python scripts/manage_runs.py delete erm-baseline-v1          # prompts for confirmation
python scripts/manage_runs.py delete erm-baseline-v1 --yes    # skips the prompt
```

| Command | Meaning |
|---|---|
| `list` | Prints every directory under `outputs/`, with last-modified time and total size. |
| `delete <run_name>` | Deletes `outputs/<run_name>/` (checkpoints, logs, everything under it) after an interactive `[y/N]` confirmation. |
| `delete <run_name> --yes` | Same, without the prompt — for scripting cleanup. |

Not a Hydra entry point (plain `argparse`, like `download_data.py`) — it
operates directly on the filesystem, not on any particular run's config.

---

## Category: Evaluation

### `scripts/evaluate.py` — evaluate an existing checkpoint

Loads a checkpoint and runs linear-probe evaluation against it, printing a
comparison to the published WILDS baseline. Separate from `train.py` so
re-evaluating doesn't require re-running training.

```bash
python scripts/evaluate.py backend=scratch \
    backend.pretrained_checkpoint=outputs/erm-baseline-v1/pretrain_checkpoint.pt \
    data=iwildcam_full
```

| Parameter | Required | Meaning |
|---|---|---|
| `backend.pretrained_checkpoint` | yes | Path to the checkpoint to evaluate. Script raises if unset. |
| `backend` | no (default `scratch`) | Which backend's encoder to load the checkpoint into. |
| `data` | no (default `iwildcam_subset`) | Which split(s) to evaluate on. |

Prints macro-F1 / accuracy per split next to the published WILDS ERM baseline
(`wildjepa.eval.baselines.PUBLISHED_BASELINES`) via `print_comparison`.

### Cross-backend correctness check

Once you have an official `facebookresearch/ijepa` checkpoint or a Hugging Face
one, this is the correctness check described in [`design.md`](design.md): load
the same weights into `scratch` and a reference backend, run the same image
through both, and diff the embeddings.

```bash
# reference backend needs the repo checked out
git clone https://github.com/facebookresearch/ijepa third_party/ijepa

python scripts/evaluate.py backend=fb_ijepa \
    backend.pretrained_checkpoint=/path/to/official_checkpoint.pth.tar
```

There's no turnkey script for the embedding-diff itself yet (see
[`roadmap.md`](roadmap.md)); `tests/test_integration_tiny_pretrain.py`'s checkpoint
round-trip test is the analogous check within the `scratch` backend alone: save a
pretrained context encoder, reload it into a fresh `ScratchEncoder`, confirm
identical outputs.

---

## Composing configs: backend / device / data / train

Any Hydra config group can be overridden independently of the others, e.g.:

```bash
python scripts/train.py backend=hf_ijepa device=cuda data=iwildcam_full train.mode=linear_probe
```

| Group | Options | Overrides |
|---|---|---|
| `backend` | `scratch` \| `fb_ijepa` \| `hf_ijepa` | `backend=<name>`, then `backend.<field>=<value>` for nested params |
| `device` | `auto` \| `cpu` \| `cuda` \| `mps` | `device=<name>` |
| `data` | `synthetic` \| `iwildcam_subset` \| `iwildcam_full` | `data=<name>`, then `data.<field>=<value>` |
| `train` | `quick_check` \| `erm_baseline` | `train=<name>`, then `train.<field>=<value>` |

Full parameter tables for every group and option: [`configuration.md`](configuration.md).
