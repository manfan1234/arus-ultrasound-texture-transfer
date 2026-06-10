"""
Loss functions for CycleGAN-based ultrasound texture transfer.

This module contains only loss definitions and texture/statistical penalties.
It does not contain models, datasets, paths, device selection or training loops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """
    Least-squares GAN loss.

    This implements the LSGAN objective commonly used in CycleGAN training.
    """

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.criterion(prediction, target)


def gradient_loss(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize differences between image gradients.

    This is useful to discourage excessive changes in anatomical edges or
    large-scale structures while allowing texture modifications.
    """

    dx_a = a[:, :, :, 1:] - a[:, :, :, :-1]
    dx_b = b[:, :, :, 1:] - b[:, :, :, :-1]

    dy_a = a[:, :, 1:, :] - a[:, :, :-1, :]
    dy_b = b[:, :, 1:, :] - b[:, :, :-1, :]

    return torch.mean(torch.abs(dx_a - dx_b)) + torch.mean(torch.abs(dy_a - dy_b))


def high_freq(
    x: torch.Tensor,
    kernel_size: int = 9,
) -> torch.Tensor:
    """
    Extract a simple high-frequency component.

    The image is decomposed as:

        high frequency = image - local average

    Parameters
    ----------
    x:
        Tensor with shape (B, C, H, W).
    kernel_size:
        Averaging kernel size. Must be odd.
    """

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    blur = F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    return x - blur


def high_freq_energy_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    kernel_size: int = 9,
) -> torch.Tensor:
    """
    Match the average high-frequency energy between fake and real images.

    This does not force pixel-wise noise correspondence. It only compares
    the global magnitude of high-frequency texture.
    """

    hf_fake = high_freq(fake, kernel_size=kernel_size)
    hf_real = high_freq(real, kernel_size=kernel_size)

    energy_fake = torch.mean(torch.abs(hf_fake))
    energy_real = torch.mean(torch.abs(hf_real))

    return torch.abs(energy_fake - energy_real)


def local_std_map(
    x: torch.Tensor,
    kernel_size: int = 7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute a local standard-deviation map.

    This captures local granular variability, useful for speckle-like texture.
    """

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    mean = F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    mean2 = F.avg_pool2d(
        x * x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    var = torch.clamp(mean2 - mean * mean, min=eps)

    return torch.sqrt(var)


def local_std_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    kernel_size: int = 7,
) -> torch.Tensor:
    """
    Match local standard-deviation maps between fake and real images.
    """

    return torch.mean(
        torch.abs(
            local_std_map(fake, kernel_size=kernel_size)
            - local_std_map(real, kernel_size=kernel_size)
        )
    )


__all__ = [
    "GANLoss",
    "gradient_loss",
    "high_freq",
    "high_freq_energy_loss",
    "local_std_map",
    "local_std_loss",
]
