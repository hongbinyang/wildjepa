"""A fully self-contained synthetic "camera-trap-like" dataset: no download,
no internet, deterministic given a seed. Exists purely so the full pipeline
(masking -> pretraining -> linear probe -> metrics) can be smoke-tested in
seconds, before spending time on the real (large, slow-to-download) WILDS
iWildCam data. It is NOT a stand-in for the real benchmark -- see
docs/design.md.

Each "species" is a distinct combination of blob shape (circle/square/cross-
like via a soft mask), color, and typical size/position jitter, rendered onto
a noisy textured background. This is easy enough that a competent encoder
should classify it well above chance, which is what makes it useful as a
pipeline sanity check -- if accuracy is near-chance, something in the
pipeline (not the model's representational capacity) is broken.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SyntheticCameraTrapDataset(Dataset):
    def __init__(
        self,
        num_classes: int = 8,
        num_images_per_class: int = 64,
        image_size: int = 224,
        seed: int = 0,
        normalize: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.num_images_per_class = num_images_per_class
        self.image_size = image_size
        self.normalize = normalize
        self.seed = seed

        # Fixed per-class appearance parameters, so "species identity" is a
        # real, learnable signal rather than pure noise.
        gen = torch.Generator().manual_seed(seed)
        self._class_colors = torch.rand(num_classes, 3, generator=gen)
        self._class_radii = torch.linspace(0.12, 0.32, num_classes)[torch.randperm(num_classes, generator=gen)]

    def __len__(self) -> int:
        return self.num_classes * self.num_images_per_class

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = index % self.num_classes
        sample_idx = index // self.num_classes
        gen = torch.Generator().manual_seed(self.seed * 100_003 + index)

        size = self.image_size
        # Noisy background, distinct per-sample so the encoder can't just
        # memorize a fixed background.
        img = 0.15 + 0.10 * torch.randn(3, size, size, generator=gen)

        color = self._class_colors[label]
        radius = self._class_radii[label].item()
        cx = 0.5 + 0.15 * (torch.rand(1, generator=gen).item() - 0.5)
        cy = 0.5 + 0.15 * (torch.rand(1, generator=gen).item() - 0.5)

        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij"
        )
        dist = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
        blob = (dist < radius).float()

        for c in range(3):
            img[c] = img[c] * (1 - blob) + color[c] * blob

        img = img.clamp(0, 1)
        if self.normalize:
            img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        return img, label
