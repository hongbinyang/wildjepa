"""Wrapper around the WILDS `iWildCam2020-WILDS` benchmark
(https://wilds.stanford.edu; Beery et al., 2020 / Koh et al., 2021).

Requires the `wilds` package (already in environment.yml) and a real,
multi-GB download on first use -- not something this project's own dev
sandbox can exercise. Use `wildjepa.data.synthetic` for no-download pipeline
smoke tests; use this module for the real benchmark numbers.

Split names, per the WILDS benchmark:
    train    -- 129,809 images from 243 camera traps
    id_val   -- held-out images, same 243 cameras (in-distribution)
    id_test  -- 8,154 images, same 243 cameras (in-distribution test)
    val      -- held-out images, different cameras (out-of-distribution)
    test     -- 42,791 images, 48 different cameras (out-of-distribution test)

For `configs/data/iwildcam_subset.yaml`, `num_species` restricts every split
to the same `num_species` most-frequent species *in the training split*
(computed once, then applied identically across splits -- otherwise val/test
could end up evaluating on species the model never saw during training,
which would silently make eval numbers meaningless). Labels are remapped to
a contiguous 0..num_species-1 range afterward, since both the sklearn linear
probe and the fine-tune classifier head expect contiguous class indices.
"""

from __future__ import annotations

import logging

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

SPLITS = ("train", "id_val", "id_test", "val", "test")


def _build_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class _RemappedSubset(Dataset):
    """Adapts a WILDS subset (which yields (x, y, metadata)) down to the
    plain (x, y) convention used elsewhere in this project, and remaps
    original species label ids to a contiguous 0..num_species-1 range."""

    def __init__(self, wilds_subset, positions: list[int], label_map: dict[int, int]):
        self.wilds_subset = wilds_subset
        self.positions = positions
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        x, y, _metadata = self.wilds_subset[self.positions[i]]
        return x, self.label_map[int(y)]


def _select_species(y_array: torch.Tensor, train_indices, num_species: int | None) -> set[int]:
    if num_species is None:
        return set(y_array[train_indices].unique().tolist())
    train_labels = y_array[train_indices]
    counts = torch.bincount(train_labels)
    top = torch.argsort(counts, descending=True)[:num_species]
    return set(top.tolist())


def _filter_and_cap(y_array: torch.Tensor, indices, keep_species: set[int], max_per_species: int | None) -> list[int]:
    labels = y_array[indices].tolist()
    kept, seen_count = [], {}
    for pos, (idx, label) in enumerate(zip(indices, labels)):
        if label not in keep_species:
            continue
        if max_per_species is not None and seen_count.get(label, 0) >= max_per_species:
            continue
        kept.append(pos)
        seen_count[label] = seen_count.get(label, 0) + 1
    return kept


def build_iwildcam_datasets(cfg):
    from wildjepa.data import DatasetBundle

    try:
        from wilds import get_dataset
    except ImportError as e:
        raise ImportError(
            "The `wilds` package is required for iWildCam data (`pip install wilds`, "
            "already listed in environment.yml). Use data=synthetic for a "
            "no-download pipeline smoke test instead."
        ) from e

    transform = _build_transform(cfg.image_size)
    dataset = get_dataset(dataset="iwildcam", download=True, root_dir=cfg.root)

    num_species = getattr(cfg, "num_species", None)
    max_per_species = getattr(cfg, "max_images_per_species", None)

    train_subset = dataset.get_subset("train", transform=transform)
    keep_species = _select_species(dataset.y_array, train_subset.indices, num_species)
    label_map = {orig: new for new, orig in enumerate(sorted(keep_species))}
    logger.info("iWildCam: keeping %d species", len(keep_species))

    out: dict[str, Dataset] = {}
    for split in SPLITS:
        wilds_subset = dataset.get_subset(split, transform=transform)
        positions = _filter_and_cap(dataset.y_array, wilds_subset.indices, keep_species, max_per_species)
        out[split] = _RemappedSubset(wilds_subset, positions, label_map)

    logger.info("iWildCam splits loaded: %s", {k: len(v) for k, v in out.items()})
    return DatasetBundle(splits=out, num_classes=len(keep_species))
