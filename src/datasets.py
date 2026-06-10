"""
Dataset and image preprocessing utilities for unpaired ultrasound image translation.

This module contains:
- image file discovery,
- default image transforms,
- unpaired X/Y dataset.

No hardcoded paths should be placed here.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


def is_image_file(path: str | Path) -> bool:
    """
    Return True if a path has a valid image extension.
    """

    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(
    directory: str | Path,
    recursive: bool = False,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> list[Path]:
    """
    List image files in a directory.

    Parameters
    ----------
    directory:
        Directory containing image files.
    recursive:
        If True, search recursively.
    extensions:
        Valid image extensions.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    valid_ext = {ext.lower() for ext in extensions}
    pattern = "**/*" if recursive else "*"

    files = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in valid_ext
    ]

    return sorted(files)


def default_transform(
    img_size: int = 256,
    grayscale: bool = True,
) -> transforms.Compose:
    """
    Default preprocessing transform.

    Images are resized to a fixed square size, converted to tensor and
    normalized to [-1, 1].

    Parameters
    ----------
    img_size:
        Output image size.
    grayscale:
        If True, convert to grayscale and replicate to 3 channels.
        If False, keep RGB input.
    """

    transform_steps: list = []

    if grayscale:
        transform_steps.append(transforms.Grayscale(num_output_channels=3))

    transform_steps.extend(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=Image.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    return transforms.Compose(transform_steps)


class UnpairedImageDataset(Dataset):
    """
    Dataset for unpaired image-to-image translation.

    Each item contains:
        {"X": image_from_domain_X, "Y": image_from_domain_Y}

    Domain X is typically synthetic ultrasound.
    Domain Y is typically real ultrasound.

    Parameters
    ----------
    dir_x:
        Directory for domain X images.
    dir_y:
        Directory for domain Y images.
    transform:
        Transform applied to both domains.
    img_size:
        Used only if transform is not provided.
    length:
        Effective dataset length. If None, uses max(len(X), len(Y)).
    random_y:
        If True, pair each X sample with a random Y sample.
    random_x:
        If True, sample X randomly as well.
    recursive:
        If True, search images recursively.
    grayscale:
        If True, convert images to grayscale replicated over 3 channels.
    """

    def __init__(
        self,
        dir_x: str | Path,
        dir_y: str | Path,
        transform: transforms.Compose | None = None,
        img_size: int = 256,
        length: int | None = None,
        random_y: bool = True,
        random_x: bool = False,
        recursive: bool = False,
        grayscale: bool = True,
    ) -> None:
        super().__init__()

        self.dir_x = Path(dir_x)
        self.dir_y = Path(dir_y)

        self.files_x = list_image_files(self.dir_x, recursive=recursive)
        self.files_y = list_image_files(self.dir_y, recursive=recursive)

        if not self.files_x:
            raise RuntimeError(f"No valid images found in domain X: {self.dir_x}")

        if not self.files_y:
            raise RuntimeError(f"No valid images found in domain Y: {self.dir_y}")

        self.transform = transform if transform is not None else default_transform(
            img_size=img_size,
            grayscale=grayscale,
        )

        self.length = length if length is not None else max(
            len(self.files_x),
            len(self.files_y),
        )

        if self.length <= 0:
            raise ValueError("length must be positive")

        self.random_y = random_y
        self.random_x = random_x

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.random_x:
            x_path = random.choice(self.files_x)
        else:
            x_path = self.files_x[idx % len(self.files_x)]

        if self.random_y:
            y_path = random.choice(self.files_y)
        else:
            y_path = self.files_y[idx % len(self.files_y)]

        x_img = Image.open(x_path).convert("RGB")
        y_img = Image.open(y_path).convert("RGB")

        x_tensor = self.transform(x_img)
        y_tensor = self.transform(y_img)

        return {
            "X": x_tensor,
            "Y": y_tensor,
        }


__all__ = [
    "IMAGE_EXTENSIONS",
    "is_image_file",
    "list_image_files",
    "default_transform",
    "UnpairedImageDataset",
]
