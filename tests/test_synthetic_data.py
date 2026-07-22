import torch

from wildjepa.data.synthetic import SyntheticCameraTrapDataset


def test_length_and_item_shapes():
    ds = SyntheticCameraTrapDataset(num_classes=4, num_images_per_class=5, image_size=32)
    assert len(ds) == 20

    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert img.dtype == torch.float32
    assert 0 <= label < 4


def test_labels_cover_all_classes():
    ds = SyntheticCameraTrapDataset(num_classes=4, num_images_per_class=5, image_size=32)
    labels = {ds[i][1] for i in range(len(ds))}
    assert labels == {0, 1, 2, 3}


def test_deterministic_given_seed():
    ds1 = SyntheticCameraTrapDataset(num_classes=3, num_images_per_class=2, image_size=32, seed=7)
    ds2 = SyntheticCameraTrapDataset(num_classes=3, num_images_per_class=2, image_size=32, seed=7)
    img1, label1 = ds1[3]
    img2, label2 = ds2[3]
    assert label1 == label2
    assert torch.equal(img1, img2)


def test_different_classes_produce_different_mean_pixel_values():
    """Sanity check that the synthetic classes are actually visually distinct
    -- if this fails, the pipeline smoke test downstream would be
    meaningless (nothing to learn)."""
    ds = SyntheticCameraTrapDataset(num_classes=4, num_images_per_class=8, image_size=64, normalize=False)
    means = {}
    for i in range(len(ds)):
        img, label = ds[i]
        means.setdefault(label, []).append(img.mean().item())
    class_means = {label: sum(vals) / len(vals) for label, vals in means.items()}
    # at least some pair of classes should differ meaningfully in mean pixel value
    values = list(class_means.values())
    assert max(values) - min(values) > 0.01
