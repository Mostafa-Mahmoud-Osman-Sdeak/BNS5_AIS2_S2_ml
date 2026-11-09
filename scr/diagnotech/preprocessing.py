"""Image transforms, dataset and loaders shared by training, evaluation and inference.

Training and inference call the *same* ``build_transforms`` function. Only the
augmentation flag differs, so preprocessing cannot silently drift between stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision import transforms as T

from .data_manifest import sampler_weights


def build_transforms(
    preprocessing: Mapping[str, Any],
    augment: bool = False,
    augmentation: Mapping[str, Any] | None = None,
) -> T.Compose:
    """Return the image pipeline: Resize -> [augmentation] -> ToTensor -> Normalize.

    Augmentation (training split only) is deliberately conservative: small rotation,
    limited translation, mild scale jitter and mild brightness/contrast change.
    Horizontal and vertical flips are excluded because they alter cardiac and
    mediastinal laterality, which is clinically meaningful.
    """
    size = int(preprocessing["image_size"])
    steps: list[Any] = [T.Resize((size, size))]

    if augment:
        if augmentation is None:
            raise ValueError("augmentation config is required when augment=True")
        a = augmentation
        translate = float(a["translate_fraction"])
        steps.extend([
            T.RandomAffine(degrees=float(a["rotation_degrees"]), translate=(translate, translate)),
            T.RandomResizedCrop(size, scale=tuple(a["rrc_scale"]), ratio=tuple(a["rrc_ratio"])),
            T.ColorJitter(brightness=float(a["brightness"]), contrast=float(a["contrast"])),
        ])

    steps.extend([
        T.ToTensor(),
        T.Normalize(mean=list(preprocessing["mean"]), std=list(preprocessing["std"])),
    ])
    return T.Compose(steps)


def load_rgb_image(source: str | Path | Image.Image) -> Image.Image:
    """Open an image and convert to 3-channel RGB. Grayscale radiographs are replicated across channels."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(source) as img:
        return img.convert("RGB")


class ChestXrayDataset(Dataset):
    """Pairs each manifest row's image with its 8-dimensional multi-hot target vector."""

    def __init__(self, manifest: pd.DataFrame, classes: Sequence[str], transform: T.Compose) -> None:
        required = {"image_path", *classes}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        self.image_paths: list[str] = manifest["image_path"].astype(str).tolist()
        self.targets: np.ndarray = manifest[list(classes)].to_numpy(dtype=np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = load_rgb_image(self.image_paths[index])
        return self.transform(image), torch.from_numpy(self.targets[index])


def build_weighted_sampler(manifest: pd.DataFrame, classes: Sequence[str], seed: int) -> WeightedRandomSampler:
    """Sampler drawing images with probability proportional to their rare-label weight (training only)."""
    weights = torch.as_tensor(sampler_weights(manifest, classes), dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler | None,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    """DataLoader with a seeded generator and persistent workers (when workers are used)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )
