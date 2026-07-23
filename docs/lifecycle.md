# Lifecycle

`usage.md` is organized by *category* (data, training, evaluation) and
`configuration.md` by *config group* (backend, data, device, train). This
document is organized by *time* -- the order a run actually goes through, what
depends on what, and how one run is told apart from another. It also records
open design questions about run management that don't have code behind them
yet, so they don't get silently re-decided differently later.

## Run identity: what identifies "a run" today

Every invocation of `scripts/train.py` or `scripts/train_erm_baseline.py` is
identified by `run_name` (`configs/config.yaml` / `configs/erm_baseline_config.yaml`),
which *is* its directory name -- `output_dir` is always `outputs/${run_name}`,
not an independent path. `run_name` defaults to a timestamp, but naming it
explicitly gives the run a stable identity that's also directly usable as a
path, with nothing to keep in sync between the two:

```bash
python scripts/train_erm_baseline.py run_name=erm-baseline-v1
# -> outputs/erm-baseline-v1/
```

That directory is where everything about the run lives:

- Hydra's own run log.
- Checkpoints, under `outputs/<run_name>/checkpoints/` (currently only
  `scripts/train_erm_baseline.py` writes per-epoch checkpoints -- see
  [Phase 4: pause / resume](#phase-4-pause--resume-a-run-in-progress) below).
- Resuming (`train.resume_from=...`) references a checkpoint path inside a
  specific run's directory -- pass the same `run_name` again so new
  checkpoints keep landing in that same directory rather than a fresh one.

There's still no registry beyond the filesystem -- no command to compare
configs across runs, only list and delete them (below). `outputs/` *is* the
list of runs, by design: a real registry (comparing configs, not just names)
is worth building once several concurrent experiments are actually running
side by side, which is the point of the Phase 4 ablation/comparison work in
`roadmap.md` -- not before.

## Controlling run length

Every training command takes `train.epochs=<N>` (see `configuration.md` for
each `train=` option's default). There's no separate "max steps" or
early-stopping control today -- length is epoch-counted only.

---

## Phase 1: Environment setup

```bash
conda env create -f environment.yml
conda activate wildjepa
pip install -e .
pytest tests/
```

Confirms the environment is sound (all 42 tests, no external data needed)
before touching real data or spending real compute. See `usage.md`.

## Phase 2: Pipeline smoke test (no download)

```bash
python scripts/train.py data=synthetic train.mode=pretrain train.epochs=5
python scripts/train.py data=synthetic train.mode=linear_probe \
    backend.pretrained_checkpoint=outputs/<run_name>/pretrain_checkpoint.pt
```

Proves the I-JEPA loop itself (masking, context/target encoders, predictor,
loss, EMA) runs and learns something above chance, using the self-contained
synthetic dataset. Nothing here is a claim about real performance -- it's a
"the wiring isn't broken" check, same thing `tests/test_integration_tiny_pretrain.py`
verifies automatically.

## Phase 3: Real data acquisition

```bash
python scripts/download_data.py --root data/iwildcam
```

One-time, ~12GB, deliberately separate from any training command (see
`usage.md`). Everything from here on depends on this having completed.

## Phase 4: Baseline validation (Phase 0 in `roadmap.md`)

```bash
python scripts/train_erm_baseline.py
```

Plain supervised ResNet-50 ERM against the full real benchmark, published
WILDS hyperparameters. This exists to validate the *eval harness itself*
(macro-F1 computation, split handling) against a known published number
before trusting any I-JEPA result built on top of it -- see
`design.md`'s evaluation-pipeline diagram. This is also the run type that
currently supports pause/resume (below); a run that takes hours is exactly
where that matters.

### Pause / resume a run in progress

- **Temporary pause**, same session, resume shortly after: `kill -STOP <pid>`
  then `kill -CONT <pid>` -- a plain OS-level freeze, no application support
  needed, works today for any script.
- **Full stop and resume later** (process killed, machine rebooted, picking
  back up in a new session): `scripts/train_erm_baseline.py` and
  `scripts/train.py train.mode=pretrain` both support this. Each checkpoints
  model + optimizer + epoch index every `train.checkpoint_every_epochs`
  epochs; resume with `train.resume_from=<output_dir>/checkpoints/<...>_latest.pt`
  (`erm_baseline_latest.pt` or `pretrain_latest.pt`). Training continues at
  the epoch *after* the one recorded in the checkpoint -- it does not redo
  completed epochs, and for `pretrain` the EMA momentum schedule (which
  depends on absolute step count, not just epoch) picks up correctly too.
  Full walkthrough: `usage.md#pausing-and-resuming-a-run`.
- `scripts/train.py`'s `finetune` mode does **not** yet support this -- it
  only saves a final checkpoint at the very end of the run, so a killed
  fine-tuning run currently loses all progress. Worth the same treatment
  once real fine-tuning runs get long enough for it to matter (Phase 7
  below).

### Monitoring a run in progress

```bash
tensorboard --logdir outputs/
```

Every training command writes loss curves, per-split macro-F1/accuracy, and
per-species F1 (for `id_test`/`test`) to `outputs/<run_name>/tensorboard/` as
it runs -- open the dashboard while a run is still going, not just after.
Full reference: `usage.md#category-monitoring`.

## Phase 5: Self-supervised pretraining

```bash
# fast iteration
python scripts/train.py data=iwildcam_subset train.mode=pretrain train.epochs=20

# full benchmark, once the subset pipeline is validated
python scripts/train.py data=iwildcam_full train.mode=pretrain train.epochs=100 \
    device=cuda data.batch_size=64
```

Scratch backend only. Mechanically verified on real Apple Silicon MPS
hardware (loss decreases cleanly, checkpoint/resume round-trips correctly),
but not yet run against real iWildCam data -- this is the next real step
after Phase 4 confirms the eval harness is trustworthy. Getting the MPS run
working surfaced a real PyTorch/MPS bug in `Conv2d`'s backward (see
`design.md`, "Honest limitations" and `models/scratch/patch_embed.py`) that
had silently never been exercised before, since the test suite only ever ran
this path on CPU.

## Phase 6: Cross-backend correctness check

```bash
git clone https://github.com/facebookresearch/ijepa third_party/ijepa
python scripts/evaluate.py backend=fb_ijepa \
    backend.pretrained_checkpoint=/path/to/official_checkpoint.pth.tar
```

Loads the same official weights into `scratch` and a reference backend, runs
the same image through both, diffs the embeddings. Not yet done -- no
checkpoint downloaded, no diff script exists yet (`usage.md`). This validates
the `scratch` implementation's correctness independent of whether pretraining
"worked" in the SSL sense.

## Phase 7: Evaluation

```bash
# frozen linear probe (standard SSL eval protocol)
python scripts/train.py train.mode=linear_probe backend=scratch \
    backend.pretrained_checkpoint=outputs/.../pretrain_checkpoint.pt data=iwildcam_full

# end-to-end fine-tune (upper bound)
python scripts/train.py data=iwildcam_full train.mode=finetune \
    backend.pretrained_checkpoint=outputs/.../pretrain_checkpoint.pt

# re-evaluate an existing checkpoint without retraining
python scripts/evaluate.py backend=scratch data=iwildcam_full \
    backend.pretrained_checkpoint=outputs/.../pretrain_checkpoint.pt
```

Compares macro-F1 (ID and OOD) against both the published ERM baseline and
(once Phase 4 finishes) this project's own reproduced ERM number.

## Phase 8: Iterate / compare

Once more than one pretraining run exists (different hyperparameters,
different backbone sizes, masking ratios, etc.), name each one
(`run_name=vit-s-mask015`, `run_name=vit-s-mask020`, ...) so comparing them
means comparing named directories, not timestamps -- and so
`tensorboard --logdir outputs/` (Phase 4) shows them side by side
automatically, curves and per-species F1 included, not just the final
number. `roadmap.md` Phase 4 ("Ablations... comparison table") is this phase
at full scale, and remains the natural point to revisit whether a real
config-comparing registry (not just names) is worth building -- see
[Run identity](#run-identity-what-identifies-a-run-today).

## Phase 9: List existing runs

```bash
python scripts/manage_runs.py list
```

Prints every directory under `outputs/` -- i.e. every `run_name` that
exists -- with last-modified time and size. Not tied to any one point in the
sequence above: useful before resuming (Phase 4) to find the right
`run_name`, before naming a new comparison run (Phase 8) to avoid colliding
with an old one, or just to see what's accumulated. Since a run's directory
name *is* its `run_name` (no separate registry to keep in sync -- see
[Run identity](#run-identity-what-identifies-a-run-today)), this is a direct
listing of `outputs/`, not a lookup against some other source of truth.

## Phase 10: Cleanup

```bash
python scripts/manage_runs.py delete <run_name>          # prompts for confirmation
python scripts/manage_runs.py delete <run_name> --yes    # for scripted cleanup
```

Checkpoints are large (a single ResNet-50 checkpoint is ~280MB; I-JEPA
pretraining checkpoints will be larger), so failed experiments, superseded
hyperparameter sweeps, and finished ablations are worth deleting once their
results are recorded elsewhere (a comparison table, this document, `roadmap.md`).
Deleting is just deleting the directory; `manage_runs.py` exists mainly so
that's a deliberate, confirmed action rather than a bare `rm -rf`.

---

Current status against these phases: `roadmap.md` is the authoritative,
up-to-date checklist -- this document describes the shape of the journey, that
one tracks exactly where things stand today.
