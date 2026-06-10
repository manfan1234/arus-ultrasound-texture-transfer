"""
Neural network architectures for CycleGAN-based ultrasound texture transfer.

This module contains only model definitions:
- ResNet generator
- Residual generator wrapper
- PatchGAN discriminator
- Local PatchGAN discriminator

No paths, no device selection, no training loop and no loss definitions
should be placed here.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


UpsampleMode = Literal["transpose", "bilinear", "nearest"]


class ResNetBlock(nn.Module):
    """
    Standard residual block used inside the ResNet generator.

    Parameters
    ----------
    dim:
        Number of feature channels.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet-based generator for image-to-image translation.

    This class supports both generator variants used in the project:

    - upsample_mode="transpose":
        Uses ConvTranspose2d upsampling. This matches the simpler baseline
        non-residual CycleGAN architecture.

    - upsample_mode="bilinear" or "nearest":
        Uses interpolation followed by convolution. This matches the more
        stable residual texture-transfer architecture.

    Parameters
    ----------
    input_nc:
        Number of input channels.
    output_nc:
        Number of output channels.
    ngf:
        Number of filters in the first convolutional layer.
    n_blocks:
        Number of residual blocks in the bottleneck.
    upsample_mode:
        Upsampling strategy: "transpose", "bilinear" or "nearest".
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int,
        n_blocks: int,
        upsample_mode: UpsampleMode = "transpose",
    ) -> None:
        super().__init__()

        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")

        if upsample_mode not in {"transpose", "bilinear", "nearest"}:
            raise ValueError(
                "upsample_mode must be one of: 'transpose', 'bilinear', 'nearest'"
            )

        layers: list[nn.Module] = []

        # Initial convolution
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2

        for _ in range(2):
            layers += [
                nn.Conv2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]

            in_features = out_features
            out_features = in_features * 2

        # Residual bottleneck
        for _ in range(n_blocks):
            layers += [ResNetBlock(in_features)]

        # Upsampling
        out_features = in_features // 2

        for _ in range(2):
            if upsample_mode == "transpose":
                layers += [
                    nn.ConvTranspose2d(
                        in_features,
                        out_features,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                ]

            else:
                if upsample_mode == "bilinear":
                    layers += [
                        nn.Upsample(
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=False,
                        )
                    ]
                elif upsample_mode == "nearest":
                    layers += [
                        nn.Upsample(
                            scale_factor=2,
                            mode="nearest",
                        )
                    ]

                layers += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(
                        in_features,
                        out_features,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                    ),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                ]

            in_features = out_features
            out_features = in_features // 2

        # Output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualWrapper(nn.Module):
    """
    Wraps a generator as a residual generator.

    Instead of returning:

        G(x)

    it returns:

        x + alpha * G(x)

    This is useful when the desired transformation should mainly modify
    image texture while preserving the input anatomy.

    Parameters
    ----------
    base_generator:
        Generator that predicts the residual image.
    alpha:
        Multiplicative factor applied to the predicted residual.
    clamp_output:
        Whether to clamp the output.
    min_value:
        Minimum output value after clamping.
    max_value:
        Maximum output value after clamping.
    """

    def __init__(
        self,
        base_generator: nn.Module,
        alpha: float,
        clamp_output: bool = True,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ) -> None:
        super().__init__()

        self.base = base_generator
        self.alpha = alpha
        self.clamp_output = clamp_output
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.base(x)
        out = x + self.alpha * residual

        if self.clamp_output:
            out = torch.clamp(out, self.min_value, self.max_value)

        return out


class PatchGANDiscriminator(nn.Module):
    """
    Standard PatchGAN discriminator.

    Parameters
    ----------
    input_nc:
        Number of input channels.
    ndf:
        Number of filters in the first convolutional layer.
    n_layers:
        Number of intermediate convolutional blocks.
    max_mult:
        Maximum channel multiplier.
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int,
        n_layers: int = 3,
        max_mult: int = 8,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        layers: list[nn.Module] = [
            nn.Conv2d(
                input_nc,
                ndf,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1

        for n in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, max_mult)
            stride = 2 if n < n_layers else 1

            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers += [
            nn.Conv2d(
                ndf * nf_mult,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LocalPatchDiscriminator(nn.Module):
    """
    Smaller discriminator focused on local texture.

    It is intended to emphasize fine-scale patterns such as speckle,
    granular texture and local high-frequency structure.

    Parameters
    ----------
    input_nc:
        Number of input channels.
    ndf:
        Number of filters in the first convolutional layer.
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                input_nc,
                ndf,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                ndf,
                ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                ndf * 2,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_generator(
    input_nc: int,
    output_nc: int,
    ngf: int,
    n_blocks: int,
    upsample_mode: UpsampleMode,
    residual: bool = False,
    alpha: float | None = None,
    clamp_output: bool = True,
) -> nn.Module:
    """
    Convenience factory for creating either a standard or residual generator.
    """

    base = ResNetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_blocks=n_blocks,
        upsample_mode=upsample_mode,
    )

    if not residual:
        return base

    if alpha is None:
        raise ValueError("alpha must be provided when residual=True")

    return ResidualWrapper(
        base_generator=base,
        alpha=alpha,
        clamp_output=clamp_output,
    )


__all__ = [
    "ResNetBlock",
    "ResNetGenerator",
    "ResidualWrapper",
    "PatchGANDiscriminator",
    "LocalPatchDiscriminator",
    "build_generator",
]
