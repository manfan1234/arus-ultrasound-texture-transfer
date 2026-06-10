"""
Inference utilities for CycleGAN-based ultrasound texture transfer.

This module contains reusable inference functions:
- build and load a generator,
- translate a single image,
- translate all images in a directory.

No hardcoded paths should be placed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.transforms import Compose

from .datasets import default_transform, list_image_files
from .models import build_generator, UpsampleMode
from .utils import (
    denormalize,
    ensure_dir,
    save_tensor_as_image,
)


def load_generator_for_inference(
    model_path: str | Path,
    input_nc: int,
    output_nc: int,
    ngf: int,
    n_blocks: int,
    upsample_mode: UpsampleMode,
    device: torch.device | str,
    residual: bool = False,
    alpha: float | None = None,
    clamp_output: bool = True,
    strict: bool = True,
) -> nn.Module:
    """
    Build a generator and load its state_dict for inference.

    Parameters
    ----------
    model_path:
        Path to a .pth checkpoint containing a generator state_dict.
    input_nc:
        Number of input channels.
    output_nc:
        Number of output channels.
    ngf:
        Number of filters in the first generator layer.
    n_blocks:
        Number of residual blocks.
    upsample_mode:
        "transpose", "bilinear" or "nearest".
    device:
        Torch device.
    residual:
        Whether the checkpoint corresponds to a ResidualWrapper generator.
    alpha:
        Residual scaling factor. Required when residual=True.
    clamp_output:
        Whether the residual wrapper clamps output to [-1, 1].
    strict:
        Strict state_dict loading.
    """

    device = torch.device(device)

    model = build_generator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_blocks=n_blocks,
        upsample_mode=upsample_mode,
        residual=residual,
        alpha=alpha,
        clamp_output=clamp_output,
    )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=strict)

    model.to(device)
    model.eval()

    return model


def translate_tensor(
    tensor: torch.Tensor,
    generator: nn.Module,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Translate a normalized tensor image using a generator.

    Parameters
    ----------
    tensor:
        Tensor with shape (C, H, W) or (1, C, H, W), normalized to [-1, 1].
    generator:
        Generator model in eval mode.
    device:
        Torch device.

    Returns
    -------
    output:
        Tensor with shape (C, H, W), normalized to [-1, 1].
    """

    device = torch.device(device)

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    if tensor.dim() != 4:
        raise ValueError(
            f"Expected tensor shape (C,H,W) or (1,C,H,W), got {tuple(tensor.shape)}"
        )

    generator.eval()

    with torch.no_grad():
        output = generator(tensor.to(device))

    return output[0].detach().cpu()


def translate_image(
    image: Image.Image,
    generator: nn.Module,
    device: torch.device | str,
    transform: Compose | None = None,
    img_size: int = 256,
    grayscale: bool = True,
) -> np.ndarray:
    """
    Translate a PIL image and return a NumPy image in [0, 1].

    Parameters
    ----------
    image:
        Input PIL image.
    generator:
        Generator model.
    device:
        Torch device.
    transform:
        Optional preprocessing transform.
    img_size:
        Used if transform is not provided.
    grayscale:
        Used if transform is not provided.
    """

    if transform is None:
        transform = default_transform(
            img_size=img_size,
            grayscale=grayscale,
        )

    image = image.convert("RGB")
    tensor = transform(image)

    output_tensor = translate_tensor(
        tensor=tensor,
        generator=generator,
        device=device,
    )

    return denormalize(output_tensor)


def translate_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    generator: nn.Module,
    device: torch.device | str,
    transform: Compose | None = None,
    img_size: int = 256,
    grayscale: bool = True,
    recursive: bool = False,
    output_suffix: str = "",
    output_extension: str | None = None,
    channel: int = 0,
) -> list[Path]:
    """
    Translate all images in a directory and save them.

    Parameters
    ----------
    input_dir:
        Directory containing input images.
    output_dir:
        Directory where translated images will be saved.
    generator:
        Generator model.
    device:
        Torch device.
    transform:
        Optional preprocessing transform.
    img_size:
        Used if transform is not provided.
    grayscale:
        Used if transform is not provided.
    recursive:
        If True, search input images recursively.
    output_suffix:
        Suffix appended to each output filename stem.
    output_extension:
        If provided, force this extension, e.g. ".png".
        If None, preserve original extension.
    channel:
        Channel extracted when saving grayscale output.

    Returns
    -------
    saved_paths:
        List of saved output image paths.
    """

    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    if transform is None:
        transform = default_transform(
            img_size=img_size,
            grayscale=grayscale,
        )

    image_paths = list_image_files(
        input_dir,
        recursive=recursive,
    )

    if not image_paths:
        raise RuntimeError(f"No valid images found in {input_dir}")

    saved_paths: list[Path] = []

    generator.eval()

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)

        output_tensor = translate_tensor(
            tensor=tensor,
            generator=generator,
            device=device,
        )

        if recursive:
            relative = image_path.relative_to(input_dir)
            out_parent = output_dir / relative.parent
            ensure_dir(out_parent)
        else:
            out_parent = output_dir

        ext = output_extension if output_extension is not None else image_path.suffix

        if not ext.startswith("."):
            ext = "." + ext

        out_name = f"{image_path.stem}{output_suffix}{ext}"
        out_path = out_parent / out_name

        save_tensor_as_image(
            tensor=output_tensor,
            output_path=out_path,
            channel=channel,
        )

        saved_paths.append(out_path)

    return saved_paths


__all__ = [
    "load_generator_for_inference",
    "translate_tensor",
    "translate_image",
    "translate_directory",
]
