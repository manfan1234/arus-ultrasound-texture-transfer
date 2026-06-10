"""
Train a baseline non-residual CycleGAN model.

This script trains a standard CycleGAN for unpaired ultrasound image translation:

    X: synthetic/source ultrasound images
    Y: real/target ultrasound images

No ARUS simulator code or protected data is required. The user must provide
two folders of images.

Example
-------
python -m scripts.train_baseline \
  --synthetic_dir /path/to/synthetic_images \
  --real_dir /path/to/real_images \
  --output_dir outputs/baseline_run \
  --epochs 20 \
  --img_size 256 \
  --batch_size 1 \
  --ngf 64 \
  --n_blocks 9 \
  --upsample_mode transpose
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import UnpairedImageDataset, default_transform
from src.models import build_generator, PatchGANDiscriminator
from src.training import (
    ImageBuffer,
    CycleGANTrainingConfig,
    train_cycle_gan,
)
from src.utils import get_device, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline non-residual CycleGAN model."
    )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    parser.add_argument(
        "--synthetic_dir",
        type=Path,
        required=True,
        help="Directory containing source/synthetic ultrasound images, domain X.",
    )

    parser.add_argument(
        "--real_dir",
        type=Path,
        required=True,
        help="Directory containing target/real ultrasound images, domain Y.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where checkpoints will be saved.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search images recursively inside synthetic_dir and real_dir.",
    )

    # ------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------

    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size used for resizing.",
    )

    parser.add_argument(
        "--no_grayscale",
        action="store_true",
        help="Do not convert images to grayscale replicated over 3 channels.",
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size.",
    )

    parser.add_argument(
        "--dataset_length",
        type=int,
        default=None,
        help=(
            "Effective dataset length per epoch. "
            "If omitted, uses max(len(synthetic), len(real))."
        ),
    )

    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum number of batches per epoch. Useful for quick tests.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--decay_epoch",
        type=int,
        default=10,
        help="Epoch from which linear learning-rate decay starts.",
    )

    parser.add_argument(
        "--buffer_size",
        type=int,
        default=50,
        help="Fake-image buffer size.",
    )

    # ------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------

    parser.add_argument(
        "--input_nc",
        type=int,
        default=3,
        help="Number of input channels.",
    )

    parser.add_argument(
        "--output_nc",
        type=int,
        default=3,
        help="Number of output channels.",
    )

    parser.add_argument(
        "--ngf",
        type=int,
        default=64,
        help="Number of generator filters in the first layer.",
    )

    parser.add_argument(
        "--n_blocks",
        type=int,
        default=9,
        help="Number of residual blocks in both generators.",
    )

    parser.add_argument(
        "--upsample_mode",
        type=str,
        choices=["transpose", "bilinear", "nearest"],
        default="transpose",
        help="Upsampling mode used in the generator.",
    )

    # ------------------------------------------------------------
    # Discriminator
    # ------------------------------------------------------------

    parser.add_argument(
        "--ndf",
        type=int,
        default=64,
        help="Number of discriminator filters in the first layer.",
    )

    # ------------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------------

    parser.add_argument(
        "--lambda_cycle",
        type=float,
        default=10.0,
        help="Cycle-consistency loss weight.",
    )

    parser.add_argument(
        "--lambda_id",
        type=float,
        default=5.0,
        help="Identity loss weight.",
    )

    parser.add_argument(
        "--w_gan_xy",
        type=float,
        default=1.0,
        help="Adversarial loss weight for G_XY.",
    )

    parser.add_argument(
        "--w_gan_yx",
        type=float,
        default=1.0,
        help="Adversarial loss weight for G_YX.",
    )

    # ------------------------------------------------------------
    # Logging/checkpoints
    # ------------------------------------------------------------

    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Interval, in epochs, for displaying samples.",
    )

    parser.add_argument(
        "--no_show_samples",
        action="store_true",
        help="Disable matplotlib sample display.",
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Interval, in epochs, for saving checkpoints.",
    )

    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="baseline_G_XY",
        help="Checkpoint filename prefix for G_XY.",
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training.",
    )

    return parser.parse_args()


def linear_decay_lambda(
    epoch: int,
    total_epochs: int,
    decay_epoch: int,
) -> float:
    """
    Linear LR decay factor.

    Keeps LR constant before decay_epoch, then linearly decays to zero.
    """

    if epoch < decay_epoch:
        return 1.0

    denom = max(1, total_epochs - decay_epoch)
    return max(0.0, 1.0 - (epoch - decay_epoch) / denom)


def main() -> None:
    args = parse_args()

    output_dir = ensure_dir(args.output_dir)
    device = get_device(prefer_cuda=not args.cpu)

    print(f"Using device: {device}")
    print(f"Synthetic/domain X: {args.synthetic_dir}")
    print(f"Real/domain Y:      {args.real_dir}")
    print(f"Output directory:   {output_dir}")

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    transform = default_transform(
        img_size=args.img_size,
        grayscale=not args.no_grayscale,
    )

    dataset = UnpairedImageDataset(
        dir_x=args.synthetic_dir,
        dir_y=args.real_dir,
        transform=transform,
        length=args.dataset_length,
        random_y=True,
        random_x=False,
        recursive=args.recursive,
        grayscale=not args.no_grayscale,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print(f"Images X: {len(dataset.files_x)}")
    print(f"Images Y: {len(dataset.files_y)}")
    print(f"Effective dataset length: {len(dataset)}")

    # ------------------------------------------------------------
    # Models
    # ------------------------------------------------------------

    G_XY = build_generator(
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        ngf=args.ngf,
        n_blocks=args.n_blocks,
        upsample_mode=args.upsample_mode,
        residual=False,
    ).to(device)

    G_YX = build_generator(
        input_nc=args.output_nc,
        output_nc=args.input_nc,
        ngf=args.ngf,
        n_blocks=args.n_blocks,
        upsample_mode=args.upsample_mode,
        residual=False,
    ).to(device)

    D_Y = PatchGANDiscriminator(
        input_nc=args.output_nc,
        ndf=args.ndf,
    ).to(device)

    D_X = PatchGANDiscriminator(
        input_nc=args.input_nc,
        ndf=args.ndf,
    ).to(device)

    # ------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------

    optimizer_G = optim.Adam(
        itertools.chain(G_XY.parameters(), G_YX.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    optimizer_D_Y = optim.Adam(
        D_Y.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    optimizer_D_X = optim.Adam(
        D_X.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    # ------------------------------------------------------------
    # Schedulers
    # ------------------------------------------------------------

    scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=lambda epoch: linear_decay_lambda(
            epoch,
            total_epochs=args.epochs,
            decay_epoch=args.decay_epoch,
        ),
    )

    scheduler_D_Y = optim.lr_scheduler.LambdaLR(
        optimizer_D_Y,
        lr_lambda=lambda epoch: linear_decay_lambda(
            epoch,
            total_epochs=args.epochs,
            decay_epoch=args.decay_epoch,
        ),
    )

    scheduler_D_X = optim.lr_scheduler.LambdaLR(
        optimizer_D_X,
        lr_lambda=lambda epoch: linear_decay_lambda(
            epoch,
            total_epochs=args.epochs,
            decay_epoch=args.decay_epoch,
        ),
    )

    # ------------------------------------------------------------
    # Buffers and config
    # ------------------------------------------------------------

    buffer_X = ImageBuffer(capacity=args.buffer_size)
    buffer_Y = ImageBuffer(capacity=args.buffer_size)

    config = CycleGANTrainingConfig(
        w_gan_xy=args.w_gan_xy,
        w_gan_yx=args.w_gan_yx,
        lambda_cycle_x=args.lambda_cycle,
        lambda_cycle_y=args.lambda_cycle,
        lambda_id_xy=args.lambda_id,
        lambda_id_yx=args.lambda_id,
        lambda_direct_x=0.0,
        lambda_gradient=0.0,
        lambda_high_freq=0.0,
        lambda_local_std=0.0,
        max_batches=args.max_batches,
        sample_interval=args.sample_interval,
        show_samples=not args.no_show_samples,
        output_dir=output_dir,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_prefix=args.checkpoint_prefix,
    )

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    history = train_cycle_gan(
        num_epochs=args.epochs,
        dataloader=dataloader,
        G_XY=G_XY,
        G_YX=G_YX,
        D_Y=D_Y,
        D_X=D_X,
        optim_G=optimizer_G,
        optim_D_Y=optimizer_D_Y,
        optim_D_X=optimizer_D_X,
        scheduler_G=scheduler_G,
        scheduler_D_Y=scheduler_D_Y,
        scheduler_D_X=scheduler_D_X,
        device=device,
        buffer_X=buffer_X,
        buffer_Y=buffer_Y,
        config=config,
    )

    print("Training finished.")
    print(f"Final checkpoint saved in: {output_dir}")
    print(f"Last epoch history: {history[-1] if history else 'No history'}")


if __name__ == "__main__":
    main()
