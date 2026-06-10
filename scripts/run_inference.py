"""
Run inference with a trained CycleGAN generator.

Example
-------
python -m scripts.run_inference \
  --model checkpoints/generator_XY_final.pth \
  --input_dir data/synthetic \
  --output_dir outputs/realistic \
  --ngf 64 \
  --n_blocks 9 \
  --upsample_mode bilinear \
  --residual \
  --alpha 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.inference import (
    load_generator_for_inference,
    translate_directory,
)
from src.datasets import default_transform
from src.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained CycleGAN generator to a folder of images."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to generator .pth checkpoint.",
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where translated images will be saved.",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size used during preprocessing.",
    )

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
        required=True,
        help="Number of generator filters in the first layer.",
    )

    parser.add_argument(
        "--n_blocks",
        type=int,
        required=True,
        help="Number of residual blocks in the generator.",
    )

    parser.add_argument(
        "--upsample_mode",
        type=str,
        choices=["transpose", "bilinear", "nearest"],
        required=True,
        help="Generator upsampling mode.",
    )

    parser.add_argument(
        "--residual",
        action="store_true",
        help="Use ResidualWrapper architecture.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Residual scaling factor. Required if --residual is used.",
    )

    parser.add_argument(
        "--no_grayscale",
        action="store_true",
        help="Do not convert images to grayscale replicated over 3 channels.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input images recursively.",
    )

    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix appended to output filenames.",
    )

    parser.add_argument(
        "--output_extension",
        type=str,
        default=".png",
        help="Output image extension, e.g. .png, .jpg, .tif.",
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Output channel used for grayscale saving.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.residual and args.alpha is None:
        raise ValueError("--alpha must be provided when using --residual")

    device = get_device(prefer_cuda=not args.cpu)

    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")

    generator = load_generator_for_inference(
        model_path=args.model,
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        ngf=args.ngf,
        n_blocks=args.n_blocks,
        upsample_mode=args.upsample_mode,
        residual=args.residual,
        alpha=args.alpha,
        device=device,
    )

    transform = default_transform(
        img_size=args.img_size,
        grayscale=not args.no_grayscale,
    )

    saved_paths = translate_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        generator=generator,
        device=device,
        transform=transform,
        recursive=args.recursive,
        output_suffix=args.output_suffix,
        output_extension=args.output_extension,
        channel=args.channel,
    )

    print(f"Saved {len(saved_paths)} translated images in: {args.output_dir}")


if __name__ == "__main__":
    main()
