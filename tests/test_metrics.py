import numpy as np

from wildjepa.eval.metrics import accuracy, few_shot_indices, macro_f1, per_class_f1


def test_macro_f1_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y_true, y_true) == 1.0


def test_macro_f1_penalizes_rare_class_errors_more_than_accuracy():
    # 9 examples of class 0 (all correct), 1 example of class 1 (wrong).
    y_true = np.array([0] * 9 + [1])
    y_pred = np.array([0] * 9 + [0])

    acc = accuracy(y_true, y_pred)
    f1 = macro_f1(y_true, y_pred)

    assert acc == 0.9
    assert f1 < acc, "macro-F1 should be pulled down by the missed rare class, unlike accuracy"


def test_per_class_f1_keys_cover_all_labels():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])
    scores = per_class_f1(y_true, y_pred)
    assert set(scores.keys()) == {0, 1, 2}
    assert scores[0] == 1.0
    assert scores[2] == 0.0  # never predicted


def test_few_shot_indices_respects_k():
    labels = np.array([0] * 20 + [1] * 5 + [2] * 1)
    idx = few_shot_indices(labels, k=3, seed=0)
    selected_labels = labels[idx]
    counts = {cls: int((selected_labels == cls).sum()) for cls in np.unique(labels)}
    assert counts[0] == 3
    assert counts[1] == 3
    assert counts[2] == 1  # class 2 only has 1 example total


def test_few_shot_indices_deterministic_given_seed():
    labels = np.array([0] * 10 + [1] * 10)
    idx1 = few_shot_indices(labels, k=2, seed=42)
    idx2 = few_shot_indices(labels, k=2, seed=42)
    assert np.array_equal(idx1, idx2)
