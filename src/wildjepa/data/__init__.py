"""Dataset construction, dispatched by `cfg.data.name`.

    synthetic         -- wildjepa.data.synthetic, no download, for pipeline smoke tests
    iwildcam_subset   -- wildjepa.data.iwildcam, stratified small-species subset
    iwildcam_full     -- wildjepa.data.iwildcam, the full WILDS benchmark
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, random_split

from wildjepa.data.collate import make_pretrain_collate_fn, make_supervised_collate_fn

__all__ = ["DatasetBundle", "build_dataset", "make_pretrain_collate_fn", "make_supervised_collate_fn"]


@dataclass
class DatasetBundle:
    splits: dict[str, Dataset]
    num_classes: int


def build_dataset(data_cfg) -> DatasetBundle:
    if data_cfg.name == "synthetic":
        from wildjepa.data.synthetic import SyntheticCameraTrapDataset

        full = SyntheticCameraTrapDataset(
            num_classes=data_cfg.num_species,
            num_images_per_class=data_cfg.images_per_species,
            image_size=data_cfg.image_size,
        )
        n = len(full)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val
        generator = torch.Generator().manual_seed(0)
        train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=generator)
        splits = {"train": train_ds, "val": val_ds, "test": test_ds}
        return DatasetBundle(splits=splits, num_classes=data_cfg.num_species)

    if data_cfg.name in {"iwildcam_subset", "iwildcam_full"}:
        from wildjepa.data.iwildcam import build_iwildcam_datasets

        return build_iwildcam_datasets(data_cfg)

    raise ValueError(f"Unknown dataset: {data_cfg.name!r}")
