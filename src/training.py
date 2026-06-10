"""
Training utilities for CycleGAN-based ultrasound texture transfer.

This module contains:
- image buffer used for discriminator stabilization,
- training configuration dataclass,
- generic CycleGAN training loop.

It does not contain hardcoded paths, dataset locations or model construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .losses import (
    GANLoss,
    gradient_loss,
    high_freq_energy_loss,
    local_std_loss,
)
from .utils import save_model, show_cycle_sample, ensure_dir


class ImageBuffer:
    """
    Buffer of previously generated images.

    This is commonly used in CycleGAN training to stabilize discriminator
    updates by mixing recent fake images with older fake images.
    """

    def __init__(self, capacity: int = 50) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")

        self.capacity = capacity
        self.buffer: list[torch.Tensor] = []

    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        """
        Add generated images to the buffer and return a batch for discriminator training.
        """

        if self.capacity == 0:
            return data

        to_return = []

        for element in data:
            element = element.unsqueeze(0)

            if len(self.buffer) < self.capacity:
                self.buffer.append(element.detach())
                to_return.append(element)

            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.capacity - 1)
                    old = self.buffer[idx].clone()
                    self.buffer[idx] = element.detach()
                    to_return.append(old)
                else:
                    to_return.append(element)

        return torch.cat(to_return, dim=0)


@dataclass
class CycleGANTrainingConfig:
    """
    Configuration for the generic CycleGAN training loop.

    All weights are explicit. Setting a weight to 0 disables that term.
    """

    # Adversarial terms
    w_gan_xy: float = 1.0
    w_gan_yx: float = 1.0
    w_gan_local_y: float = 1.0

    # Cycle consistency
    lambda_cycle_x: float = 10.0
    lambda_cycle_y: float = 10.0

    # Identity losses
    lambda_id_xy: float = 5.0
    lambda_id_yx: float = 5.0

    # Anatomical preservation / direct regularization
    lambda_direct_x: float = 0.0
    lambda_gradient: float = 0.0

    # Texture losses
    lambda_high_freq: float = 0.0
    lambda_local_std: float = 0.0

    # Texture-loss kernels
    high_freq_kernel_size: int = 9
    local_std_kernel_size: int = 7

    # Training control
    max_batches: int | None = None
    sample_interval: int = 1
    show_samples: bool = True

    # Checkpointing
    output_dir: str | Path | None = None
    checkpoint_interval: int | None = None
    checkpoint_prefix: str = "generator_XY"


def train_cycle_gan(
    num_epochs: int,
    dataloader: DataLoader,
    G_XY: nn.Module,
    G_YX: nn.Module,
    D_Y: nn.Module,
    D_X: nn.Module,
    optim_G: optim.Optimizer,
    optim_D_Y: optim.Optimizer,
    optim_D_X: optim.Optimizer,
    device: torch.device | str,
    scheduler_G: optim.lr_scheduler.LRScheduler | None = None,
    scheduler_D_Y: optim.lr_scheduler.LRScheduler | None = None,
    scheduler_D_X: optim.lr_scheduler.LRScheduler | None = None,
    buffer_X: ImageBuffer | None = None,
    buffer_Y: ImageBuffer | None = None,
    config: CycleGANTrainingConfig | None = None,
    D_Y_local: nn.Module | None = None,
    optim_D_Y_local: optim.Optimizer | None = None,
    scheduler_D_Y_local: optim.lr_scheduler.LRScheduler | None = None,
) -> list[dict[str, float]]:
    """
    Train a CycleGAN model.

    Parameters
    ----------
    num_epochs:
        Number of training epochs.
    dataloader:
        Dataloader returning batches with keys "X" and "Y".
    G_XY:
        Generator from synthetic/source domain X to real/target domain Y.
    G_YX:
        Generator from real/target domain Y to synthetic/source domain X.
    D_Y:
        Discriminator for domain Y.
    D_X:
        Discriminator for domain X.
    optim_G:
        Optimizer for both generators.
    optim_D_Y:
        Optimizer for discriminator Y.
    optim_D_X:
        Optimizer for discriminator X.
    device:
        Torch device.
    scheduler_*:
        Optional learning-rate schedulers.
    buffer_X, buffer_Y:
        Optional fake-image buffers.
    config:
        Training configuration.
    D_Y_local:
        Optional local discriminator for fine texture in domain Y.
    optim_D_Y_local:
        Optimizer for the local discriminator.

    Returns
    -------
    history:
        List of dictionaries with epoch-level losses.
    """

    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    if config is None:
        config = CycleGANTrainingConfig()

    if D_Y_local is not None and optim_D_Y_local is None:
        raise ValueError("optim_D_Y_local must be provided when D_Y_local is used")

    device = torch.device(device)

    G_XY.to(device)
    G_YX.to(device)
    D_Y.to(device)
    D_X.to(device)

    if D_Y_local is not None:
        D_Y_local.to(device)

    criterion_gan = GANLoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)

    output_dir = None
    if config.output_dir is not None:
        output_dir = ensure_dir(config.output_dir)

    history: list[dict[str, float]] = []

    for epoch in range(num_epochs):
        G_XY.train()
        G_YX.train()
        D_Y.train()
        D_X.train()

        if D_Y_local is not None:
            D_Y_local.train()

        last_losses: dict[str, float] = {}

        for batch_idx, batch in enumerate(dataloader):
            if config.max_batches is not None and batch_idx >= config.max_batches:
                break

            real_X = batch["X"].to(device)
            real_Y = batch["Y"].to(device)

            # ============================================================
            # Train generators
            # ============================================================

            optim_G.zero_grad(set_to_none=True)

            fake_Y = G_XY(real_X)
            fake_X = G_YX(real_Y)

            rec_X = G_YX(fake_Y)
            rec_Y = G_XY(fake_X)

            id_Y = G_XY(real_Y)
            id_X = G_YX(real_X)

            loss_gan_xy_global = criterion_gan(D_Y(fake_Y), True)

            if D_Y_local is not None:
                loss_gan_xy_local = criterion_gan(D_Y_local(fake_Y), True)
                loss_gan_xy = config.w_gan_xy * (
                    loss_gan_xy_global
                    + config.w_gan_local_y * loss_gan_xy_local
                )
            else:
                loss_gan_xy_local = torch.tensor(0.0, device=device)
                loss_gan_xy = config.w_gan_xy * loss_gan_xy_global

            loss_gan_yx = config.w_gan_yx * criterion_gan(D_X(fake_X), True)

            loss_cycle_x = (
                criterion_cycle(rec_X, real_X)
                * config.lambda_cycle_x
            )
            loss_cycle_y = (
                criterion_cycle(rec_Y, real_Y)
                * config.lambda_cycle_y
            )

            loss_id_xy = (
                criterion_identity(id_Y, real_Y)
                * config.lambda_id_xy
            )
            loss_id_yx = (
                criterion_identity(id_X, real_X)
                * config.lambda_id_yx
            )

            if config.lambda_direct_x > 0:
                loss_direct_x = (
                    criterion_identity(fake_Y, real_X)
                    * config.lambda_direct_x
                )
            else:
                loss_direct_x = torch.tensor(0.0, device=device)

            if config.lambda_gradient > 0:
                loss_gradient = (
                    gradient_loss(fake_Y, real_X)
                    * config.lambda_gradient
                )
            else:
                loss_gradient = torch.tensor(0.0, device=device)

            if config.lambda_high_freq > 0:
                loss_high_freq = (
                    high_freq_energy_loss(
                        fake_Y,
                        real_Y,
                        kernel_size=config.high_freq_kernel_size,
                    )
                    * config.lambda_high_freq
                )
            else:
                loss_high_freq = torch.tensor(0.0, device=device)

            if config.lambda_local_std > 0:
                loss_local_std_value = (
                    local_std_loss(
                        fake_Y,
                        real_Y,
                        kernel_size=config.local_std_kernel_size,
                    )
                    * config.lambda_local_std
                )
            else:
                loss_local_std_value = torch.tensor(0.0, device=device)

            loss_G = (
                loss_gan_xy
                + loss_gan_yx
                + loss_cycle_x
                + loss_cycle_y
                + loss_id_xy
                + loss_id_yx
                + loss_direct_x
                + loss_gradient
                + loss_high_freq
                + loss_local_std_value
            )

            loss_G.backward()
            optim_G.step()

            # ============================================================
            # Train discriminator Y
            # ============================================================

            optim_D_Y.zero_grad(set_to_none=True)

            loss_D_Y_real = criterion_gan(D_Y(real_Y), True)

            if buffer_Y is not None:
                fake_Y_for_D = buffer_Y.push_and_pop(fake_Y.detach())
            else:
                fake_Y_for_D = fake_Y.detach()

            loss_D_Y_fake = criterion_gan(D_Y(fake_Y_for_D), False)

            loss_D_Y = 0.5 * (loss_D_Y_real + loss_D_Y_fake)
            loss_D_Y.backward()
            optim_D_Y.step()

            # ============================================================
            # Train optional local discriminator Y
            # ============================================================

            if D_Y_local is not None and optim_D_Y_local is not None:
                optim_D_Y_local.zero_grad(set_to_none=True)

                loss_D_Y_local_real = criterion_gan(D_Y_local(real_Y), True)
                loss_D_Y_local_fake = criterion_gan(
                    D_Y_local(fake_Y.detach()),
                    False,
                )

                loss_D_Y_local = 0.5 * (
                    loss_D_Y_local_real + loss_D_Y_local_fake
                )

                loss_D_Y_local.backward()
                optim_D_Y_local.step()
            else:
                loss_D_Y_local = torch.tensor(0.0, device=device)

            # ============================================================
            # Train discriminator X
            # ============================================================

            optim_D_X.zero_grad(set_to_none=True)

            loss_D_X_real = criterion_gan(D_X(real_X), True)

            if buffer_X is not None:
                fake_X_for_D = buffer_X.push_and_pop(fake_X.detach())
            else:
                fake_X_for_D = fake_X.detach()

            loss_D_X_fake = criterion_gan(D_X(fake_X_for_D), False)

            loss_D_X = 0.5 * (loss_D_X_real + loss_D_X_fake)
            loss_D_X.backward()
            optim_D_X.step()

            last_losses = {
                "loss_G": float(loss_G.detach().cpu()),
                "loss_D_Y": float(loss_D_Y.detach().cpu()),
                "loss_D_X": float(loss_D_X.detach().cpu()),
                "loss_D_Y_local": float(loss_D_Y_local.detach().cpu()),
                "loss_gan_xy": float(loss_gan_xy.detach().cpu()),
                "loss_gan_yx": float(loss_gan_yx.detach().cpu()),
                "loss_cycle_x": float(loss_cycle_x.detach().cpu()),
                "loss_cycle_y": float(loss_cycle_y.detach().cpu()),
                "loss_id_xy": float(loss_id_xy.detach().cpu()),
                "loss_id_yx": float(loss_id_yx.detach().cpu()),
                "loss_direct_x": float(loss_direct_x.detach().cpu()),
                "loss_gradient": float(loss_gradient.detach().cpu()),
                "loss_high_freq": float(loss_high_freq.detach().cpu()),
                "loss_local_std": float(loss_local_std_value.detach().cpu()),
            }

        if scheduler_G is not None:
            scheduler_G.step()

        if scheduler_D_Y is not None:
            scheduler_D_Y.step()

        if scheduler_D_X is not None:
            scheduler_D_X.step()

        if scheduler_D_Y_local is not None:
            scheduler_D_Y_local.step()

        epoch_info = {"epoch": float(epoch + 1), **last_losses}
        history.append(epoch_info)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"G: {last_losses.get('loss_G', float('nan')):.4f} | "
            f"D_Y: {last_losses.get('loss_D_Y', float('nan')):.4f} | "
            f"D_X: {last_losses.get('loss_D_X', float('nan')):.4f}",
            flush=True,
        )

        if (
            config.show_samples
            and config.sample_interval > 0
            and (epoch + 1) % config.sample_interval == 0
            and "real_X" in locals()
        ):
            G_XY.eval()
            G_YX.eval()

            with torch.no_grad():
                show_cycle_sample(
                    real_x=real_X,
                    real_y=real_Y,
                    fake_y=fake_Y,
                    fake_x=fake_X,
                    rec_x=rec_X,
                    rec_y=rec_Y,
                )

        if (
            output_dir is not None
            and config.checkpoint_interval is not None
            and config.checkpoint_interval > 0
            and (epoch + 1) % config.checkpoint_interval == 0
        ):
            save_model(
                G_XY,
                output_dir / f"{config.checkpoint_prefix}_epoch_{epoch + 1:03d}.pth",
            )

    if output_dir is not None:
        save_model(
            G_XY,
            output_dir / f"{config.checkpoint_prefix}_final.pth",
        )

    return history


__all__ = [
    "ImageBuffer",
    "CycleGANTrainingConfig",
    "train_cycle_gan",
]
