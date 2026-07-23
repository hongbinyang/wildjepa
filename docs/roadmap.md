# Roadmap

Status key: `[ ]` not started · `[~]` scaffolded/stubbed · `[x]` done

## Phase 0 — Scaffold

- [x] Repo structure, conda env, Hydra config system, git init
- [x] Backend interface (`JEPAEncoder`/`JEPAPredictor`) + factory
- [x] Stub `scratch` / `fb_ijepa` / `hf_ijepa` backends (raise `NotImplementedError`)
- [x] Baseline sanity check: reproduce published WILDS ERM macro-F1 with a plain
      supervised ResNet-50, to trust the eval harness before building on top of it.
      Full 12-epoch run against the real benchmark complete: id_test macro-F1 0.374
      (published 0.47), test (OOD) macro-F1 0.274 (published 0.33) -- right order of
      magnitude, ID > OOD as expected. Eval harness trusted; see design.md "Honest
      limitations" for the full reasoning.

## Phase 1 — Quick feasibility check (current target)

- [x] Implement `wildjepa.data`: WILDS `iWildCam2020-WILDS` wrapper, stratified small-species
      subset per `configs/data/iwildcam_subset.yaml`, multi-block masking collator, plus a
      no-download synthetic dataset for pipeline smoke tests
- [x] Implement `ScratchEncoder`/`ScratchPredictor`/`IJEPA` (ViT-S/16-capable, per
      `configs/backend/scratch.yaml`) -- full ViT, masking, predictor, EMA, loss
- [x] Implement `wildjepa.eval`: macro-F1, per-class F1, few-shot index sampling, published
      baseline comparison table
- [x] Implement `train/pretrain.py`, `train/linear_probe.py`, `train/finetune.py`
- [x] Implement real (non-stub) `fb_ijepa`/`hf_ijepa` adapters
- [x] Full test suite incl. a tiny end-to-end pretraining integration test
- [x] **Run `pytest tests/` on real hardware** -- 42/42 passing on the real M2 Mac.
- [x] Download the real `iWildCam2020-WILDS` data (`scripts/download_data.py`) --
      done, ~12GB at `data/iwildcam`.
- [x] Fixed a real bug surfaced by running on real hardware: `make_pretrain_collate_fn`/
      `make_supervised_collate_fn` returned local closures, unpicklable by
      `multiprocessing`'s spawn context (macOS/Windows default) -- broke
      `DataLoader(num_workers>0)`. Converted to module-level callable classes.
- [x] Added run identity (`run_name`, doubling as the output directory name),
      `scripts/manage_runs.py` (list/delete), checkpoint + resume support for
      `erm_baseline` and `pretrain` (not yet `finetune`/`linear_probe`), and
      TensorBoard metrics logging (loss curves, per-split macro-F1/accuracy,
      per-species F1) across all four training modes -- see `lifecycle.md`.
- [x] Fixed a real PyTorch/MPS bug surfaced by running `pretrain` on real Apple
      Silicon hardware for the first time: `Conv2d`'s MPS backward broke whenever
      its output fed a further op after a transpose (exactly what patch embedding
      always does next). Never caught before since the test suite only ran this
      path on CPU. See `design.md` "Honest limitations" and
      `models/scratch/patch_embed.py`.
- [x] Closed the test-coverage gap that let the bug above go unnoticed: added
      device-aware backward tests (`test_vit.py`, `test_integration_tiny_pretrain.py`)
      that run on `resolve_device("auto")`'s actual result instead of assuming CPU --
      safe in any environment since a GPU-less CI runner just re-exercises `cpu`.
- [x] Real pretrain run on the subset (5 epochs, real iWildCam data): loss 0.11 -> 0.05,
      mechanically clean, checkpoint saved correctly.
- [x] First linear-probe numbers on the real iWildCam subset (proof the `linear_probe`
      pipeline works on real data, first real run of that code path): 0.253 OOD macro-F1
      on the 8-species subset. Not comparable to the ERM baseline's full-182-species
      number -- different class count -- but a real, above-chance signal.
- [x] Fixed two more real bugs found chasing a "pretraining is catastrophically slow"
      symptom on the full benchmark: (1) `MultiBlockMaskCollator` resampled block size
      per batch, not just position, forcing MPS to recompile its graph almost every
      step -- fixed to sample size once per collator instance (kept, but not the
      dominant cause); (2) the actual dominant cause -- `iwildcam_full`'s default
      `batch_size=64` exceeded this memory-constrained M2 Mac's capacity (unified
      memory competing with other running apps). Measured `64` as wildly unstable
      (3-137s/step) vs `32` as fast and stable; changed the default. See `design.md`
      "Honest limitations" and `configs/data/iwildcam_full.yaml`.
- [ ] Cross-check: load a released `facebookresearch/ijepa` checkpoint into `ScratchEncoder`,
      diff output embeddings against `fb_ijepa`/`hf_ijepa` backends on identical inputs (the
      within-scratch checkpoint round-trip test exists; the cross-backend one needs an actual
      downloaded checkpoint, which needs step above first)
- [ ] Pretrain against the full 182-species benchmark (blocked until now by the
      batch-size issue above; unblocked, not yet run to completion)

## Phase 2 — Domain-adaptive pretraining at scale

- [ ] Continue I-JEPA pretraining on the *full* unlabeled `iWildCam2020-WILDS` pool
      (and optionally additional LILA BC camera-trap data)
- [ ] Move training off the M2 to cloud GPU (the device abstraction already supports this —
      no code changes expected, just `device=cuda` + a bigger `data=iwildcam_full`)
- [ ] Full linear-probe + few-shot fine-tune evaluation, ID and OOD, vs. published ERM baseline

## Phase 3 — Same-data SSL control

- [ ] Train a DINOv2 or MAE baseline on the *same* unlabeled pretraining pool, same eval
      protocol — isolates whether any win is coming from self-supervision in general or from
      I-JEPA's masked-latent-prediction objective specifically (deferred by choice from Phase 1;
      see `design.md`)

## Phase 4 — Write-up

- [ ] Ablations: masking ratio/scale, ViT size, momentum schedule
- [ ] Comparison table: ERM / DINOv2 / MAE / I-JEPA, ID + OOD macro-F1, few-shot curves
- [ ] Decide whether results warrant a public write-up (blog post, arXiv note, or just an
      internal report)

## Open questions to revisit

- Does continued pretraining need the full `iWildCam` pool, or does adding other LILA BC
  datasets materially help transfer to genuinely new species/locations?
- Is ViT-S/16 sufficient, or does the label-efficiency gap only show up at ViT-B/L scale?
- Cloud compute budget/provider once we leave the M2 (Phase 2) — not yet decided.
