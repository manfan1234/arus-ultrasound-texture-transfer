"""
General utilities for CycleGAN-based ultrasound texture transfer.

This module contains generic helpers:
- device selection,
- tensor denormalization,
- model checkpoint saving/loading,
- image saving,
- visualization.

No hardcoded project paths should be placed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Return CUDA device if available and requested, otherwise CPU.
    """

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return it as Path.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor normalized to [-1, 1] into a NumPy image in [0, 1].

    Expected tensor shape:
        (C, H, W)
    """

    img = tensor.detach().cpu().clone()
    img = img * 0.5 + 0.5
    img = torch.clamp(img, 0.0, 1.0)

    return img.permute(1, 2, 0).numpy()


def tensor_to_uint8_image(
    tensor: torch.Tensor,
    channel: int = 0,
) -> np.ndarray:
    """
    Convert a normalized tensor image to uint8.

    Parameters
    ----------
    tensor:
        Tensor with shape (C, H, W), normalized to [-1, 1].
    channel:
        Channel to extract for grayscale saving.
    """

    img = denormalize(tensor)

    if img.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, C), got {img.shape}")

    if channel < 0 or channel >= img.shape[-1]:
        raise ValueError(f"Invalid channel {channel} for image with shape {img.shape}")

    return (img[..., channel] * 255.0).clip(0, 255).astype("uint8")


def save_tensor_as_image(
    tensor: torch.Tensor,
    output_path: str | Path,
    channel: int = 0,
) -> None:
    """
    Save a normalized tensor image as a grayscale PNG/JPG/TIFF image.
    """

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    img_uint8 = tensor_to_uint8_image(tensor, channel=channel)
    Image.fromarray(img_uint8, mode="L").save(output_path)


def save_model(
    model: nn.Module,
    path: str | Path,
) -> None:
    """
    Save model state_dict.
    """

    path = Path(path)
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)


def load_state_dict(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str,
    strict: bool = True,
) -> nn.Module:
    """
    Load a state_dict into an already constructed model.
    """

    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()

    return model


def show_cycle_sample(
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    fake_y: torch.Tensor,
    fake_x: torch.Tensor,
    rec_x: torch.Tensor,
    rec_y: torch.Tensor,
    max_images: int = 1,
) -> None:
    """
    Display CycleGAN samples.

    Expected tensors have shape:
        (B, C, H, W)
    """

    real_x = real_x[:max_images].detach().cpu()
    real_y = real_y[:max_images].detach().cpu()
    fake_y = fake_y[:max_images].detach().cpu()
    fake_x = fake_x[:max_images].detach().cpu()
    rec_x = rec_x[:max_images].detach().cpu()
    rec_y = rec_y[:max_images].detach().cpu()

    titles = [
        "Synthetic X",
        "X to Y",
        "X to Y to X",
        "Real Y",
        "Y to X",
        "Y to X to Y",
    ]

    images = [
        real_x,
        fake_y,
        rec_x,
        real_y,
        fake_x,
        rec_y,
    ]

    plt.figure(figsize=(12, 8))

    for i, image_batch in enumerate(images):
        img_np = denormalize(image_batch[0])

        plt.subplot(2, 3, i + 1)
        plt.imshow(img_np[..., 0], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


__all__ = [
    "get_device",
    "ensure_dir",
    "denormalize",
    "tensor_to_uint8_image",
    "save_tensor_as_image",
    "save_model",
    "load_state_dict",
    "show_cycle_sample",
]
