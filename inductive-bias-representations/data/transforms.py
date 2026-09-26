from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def canonical_tensor(image, image_size: int = 224) -> torch.Tensor:
    return TF.resize(TF.to_tensor(image.convert("RGB")), [image_size, image_size], antialias=True)


def grayscale(x: torch.Tensor) -> torch.Tensor:
    return TF.rgb_to_grayscale(x, num_output_channels=3)


def hue_rotate(x: torch.Tensor, factor: float) -> torch.Tensor:
    return TF.adjust_hue(x, factor)


def translate_reflect(x: torch.Tensor, delta: int, direction: str) -> torch.Tensor:
    if delta == 0:
        return x.clone()
    padded = F.pad(x.unsqueeze(0), (delta, delta, delta, delta), mode="reflect").squeeze(0)
    height, width = x.shape[-2:]
    offsets = {
        "right": (delta, 0), "left": (delta, 2 * delta),
        "down": (0, delta), "up": (2 * delta, delta),
    }
    top, left = offsets[direction]
    return padded[:, top:top + height, left:left + width]


def patch_shuffle(x: torch.Tensor, official_index: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed + int(official_index))
    permutation = rng.permutation(16)
    while np.array_equal(permutation, np.arange(16)):
        permutation = rng.permutation(16)
    rows = torch.chunk(x, 4, dim=1)
    patches = [patch for row in rows for patch in torch.chunk(row, 4, dim=2)]
    shuffled = [patches[index] for index in permutation]
    return torch.cat([torch.cat(shuffled[4 * row:4 * row + 4], dim=2) for row in range(4)], dim=1)


class IndexedSTL(Dataset):
    def __init__(self, base, indices, cfg, mode="clean", delta=0, direction="right"):
        self.base, self.indices, self.cfg = base, np.asarray(indices), cfg
        self.mode, self.delta, self.direction = mode, delta, direction

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        official_index = int(self.indices[position])
        image, label = self.base[official_index]
        x = canonical_tensor(image, self.cfg.image_size)
        if self.mode == "grayscale":
            x = grayscale(x)
        elif self.mode == "hue":
            x = hue_rotate(x, self.cfg.hue_factor)
        elif self.mode == "patch":
            x = patch_shuffle(x, official_index, self.cfg.seed)
        elif self.mode == "translation":
            x = translate_reflect(x, self.delta, self.direction)
        elif self.mode != "clean":
            raise ValueError(f"Unknown intervention: {self.mode}")
        return x, int(label), official_index
