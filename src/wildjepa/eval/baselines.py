"""Published reference numbers to compare our results against, so a run's
output is legible without cross-referencing a paper by hand.

Sources (see docs/design.md for full citations):
  - WILDS: A Benchmark of in-the-Wild Distribution Shifts (Koh et al., 2021,
    arXiv:2012.07421) -- ERM (supervised ResNet-50) macro-F1 on
    iWildCam2020-WILDS.
  - Norouzzadeh et al., 2018, PNAS -- ensemble CNN on a curated, balanced
    Snapshot Serengeti species subset (accuracy, not macro-F1; not directly
    comparable to the OOD numbers above, included for context only).
"""

from __future__ import annotations

PUBLISHED_BASELINES = {
    "wilds_erm_resnet50_iwildcam_id_macro_f1": 0.47,
    "wilds_erm_resnet50_iwildcam_ood_macro_f1": 0.33,  # published range ~0.31-0.35
    "snapshot_serengeti_cnn_balanced_accuracy": 0.938,  # not macro-F1, not OOD -- context only
}

# Maps a results dict key (produced by train/linear_probe.py or
# train/finetune.py, named after the dataset split it came from) to the
# published baseline it's actually comparable to. iWildCam-WILDS's "test"
# split is the out-of-distribution test set (different cameras); "id_test"
# is the in-distribution one -- see docs/design.md.
_SPLIT_TO_BASELINE = {
    "test_macro_f1": "wilds_erm_resnet50_iwildcam_ood_macro_f1",
    "id_test_macro_f1": "wilds_erm_resnet50_iwildcam_id_macro_f1",
}


def print_comparison(results: dict[str, float]) -> None:
    """Prints `results` next to the published baselines they're most
    comparable to, for a quick sanity read after a run. Metrics with no
    obviously comparable published baseline (val splits, accuracy, synthetic
    data) just print with no comparison column filled in."""
    print(f"{'metric':45s} {'ours':>10s} {'published':>10s}")
    print("-" * 68)
    for key, ours in sorted(results.items()):
        baseline_key = _SPLIT_TO_BASELINE.get(key)
        published = PUBLISHED_BASELINES.get(baseline_key) if baseline_key else None
        published_str = f"{published:.3f}" if published is not None else "n/a"
        print(f"{key:45s} {ours:10.3f} {published_str:>10s}")
