"""
Smoke test for the repository.

This script checks that the src/ package works end-to-end:
1. creates dummy synthetic and real ultrasound-like images,
2. loads them with UnpairedImageDataset,
3. builds small CycleGAN models,
4. trains for one batch,
5. saves a checkpoint,
6. runs directory inference.
"""

from pathlib import Path
import itertools

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.datasets import UnpairedImageDataset, default_transform
from src.models import (
    build_generator,
    PatchGANDiscriminator,
    LocalPatchDiscriminator,
)
from src.training import (
    ImageBuffer,
    CycleGANTrainingConfig,
    train_cycle_gan,
)
from src.inference import (
    load_generator_for_inference,
    translate_directory,
)
from src.utils import get_device, ensure_dir


def make_dummy_images(output_dir: Path, n: int, kind: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = 256, 256

    yy, xx = np.mgrid[0:h, 0:w]

    for i in range(n):
        if kind == "synthetic":
            img = (
                80
                + 80 * np.exp(-((xx - 128) ** 2 + (yy - 130) ** 2) / (2 * 45**2))
                + 35 * np.exp(-((xx - 90) ** 2 + (yy - 95) ** 2) / (2 * 20**2))
            )
        elif kind == "real":
            rng = np.random.default_rng(seed=i)
            speckle = rng.normal(0, 25, size=(h, w))
            img = (
                70
                + 60 * np.exp(-((xx - 125) ** 2 + (yy - 135) ** 2) / (2 * 55**2))
                + speckle
            )
        else:
            raise ValueError(f"Unknown kind: {kind}")

        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(output_dir / f"{kind}_{i:03d}.png")


def main() -> None:
    root = Path("_smoke_test")
    synthetic_dir = root / "synthetic"
    real_dir = root / "real"
    output_dir = root / "outputs"
    checkpoint_dir = root / "checkpoints"

    ensure_dir(root)
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)

    make_dummy_images(synthetic_dir, n=4, kind="synthetic")
    make_dummy_images(real_dir, n=4, kind="real")

    device = get_device()
    print(f"Using device: {device}")

    transform = default_transform(img_size=128, grayscale=True)

    dataset = UnpairedImageDataset(
        dir_x=synthetic_dir,
        dir_y=real_dir,
        transform=transform,
        length=4,
        random_y=True,
        random_x=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Small models for fast testing.
    G_XY = build_generator(
        input_nc=3,
        output_nc=3,
        ngf=8,
        n_blocks=1,
        upsample_mode="bilinear",
        residual=True,
        alpha=0.2,
    )

    G_YX = build_generator(
        input_nc=3,
        output_nc=3,
        ngf=8,
        n_blocks=1,
        upsample_mode="bilinear",
        residual=False,
    )

    D_Y = PatchGANDiscriminator(input_nc=3, ndf=8)
    D_X = PatchGANDiscriminator(input_nc=3, ndf=8)
    D_Y_local = LocalPatchDiscriminator(input_nc=3, ndf=8)

    G_XY.to(device)
    G_YX.to(device)
    D_Y.to(device)
    D_X.to(device)
    D_Y_local.to(device)

    optimizer_G = optim.Adam(
        itertools.chain(G_XY.parameters(), G_YX.parameters()),
        lr=1e-4,
        betas=(0.5, 0.999),
    )

    optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D_X = optim.Adam(D_X.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D_Y_local = optim.Adam(
        D_Y_local.parameters(),
        lr=1e-4,
        betas=(0.5, 0.999),
    )

    config = CycleGANTrainingConfig(
        w_gan_xy=1.0,
        w_gan_yx=0.2,
        lambda_cycle_x=1.0,
        lambda_cycle_y=1.0,
        lambda_id_xy=0.5,
        lambda_id_yx=0.5,
        lambda_direct_x=0.1,
        lambda_gradient=0.1,
        lambda_high_freq=0.1,
        lambda_local_std=0.1,
        max_batches=1,
        sample_interval=1,
        show_samples=False,
        output_dir=checkpoint_dir,
        checkpoint_interval=1,
        checkpoint_prefix="smoke_G_XY",
    )

    history = train_cycle_gan(
        num_epochs=1,
        dataloader=dataloader,
        G_XY=G_XY,
        G_YX=G_YX,
        D_Y=D_Y,
        D_X=D_X,
        optim_G=optimizer_G,
        optim_D_Y=optimizer_D_Y,
        optim_D_X=optimizer_D_X,
        device=device,
        buffer_X=ImageBuffer(capacity=2),
        buffer_Y=ImageBuffer(capacity=2),
        config=config,
        D_Y_local=D_Y_local,
        optim_D_Y_local=optimizer_D_Y_local,
    )

    print("Training history:")
    print(history)

    checkpoint_path = checkpoint_dir / "smoke_G_XY_final.pth"

    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint was not created: {checkpoint_path}")

    generator = load_generator_for_inference(
        model_path=checkpoint_path,
        input_nc=3,
        output_nc=3,
        ngf=8,
        n_blocks=1,
        upsample_mode="bilinear",
        residual=True,
        alpha=0.2,
        device=device,
    )

    saved = translate_directory(
        input_dir=synthetic_dir,
        output_dir=output_dir,
        generator=generator,
        device=device,
        transform=transform,
        output_extension=".png",
    )

    print(f"Inference saved {len(saved)} images in {output_dir}")
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
