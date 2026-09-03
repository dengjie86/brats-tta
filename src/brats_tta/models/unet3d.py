from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    """Conv3d -> InstanceNorm3d -> LeakyReLU used by the source model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        eps: float = 1e-5,
        negative_slope: float = 1e-2,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.InstanceNorm3d(
                out_channels,
                eps=eps,
                affine=True,
                track_running_stats=False,
            ),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )


class EncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        num_convs: int,
        norm_eps: float,
        negative_slope: float,
    ) -> None:
        if num_convs < 1:
            raise ValueError("num_convs must be at least 1")
        blocks: list[nn.Module] = [
            ConvNormAct(
                in_channels,
                out_channels,
                stride=stride,
                eps=norm_eps,
                negative_slope=negative_slope,
            )
        ]
        blocks.extend(
            ConvNormAct(
                out_channels,
                out_channels,
                eps=norm_eps,
                negative_slope=negative_slope,
            )
            for _ in range(num_convs - 1)
        )
        super().__init__(*blocks)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        num_convs: int,
        norm_eps: float,
        negative_slope: float,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.convs = EncoderStage(
            out_channels + skip_channels,
            out_channels,
            stride=1,
            num_convs=num_convs,
            norm_eps=norm_eps,
            negative_slope=negative_slope,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            raise RuntimeError(
                "Decoder and skip shapes do not match. Input spatial dimensions must "
                f"be divisible by the network downsampling factor; got {x.shape[2:]} and {skip.shape[2:]}."
            )
        return self.convs(torch.cat((x, skip), dim=1))


class PlainUNet3D(nn.Module):
    """Five-stage nnU-Net-style 3D U-Net for overlapping BraTS regions.

    The default architecture is the source model agreed for this project:

    - input: four MRI modalities
    - encoder features: 32, 64, 128, 256, 320
    - two 3x3x3 convolutions per encoder and decoder stage
    - stride-2 convolutional downsampling
    - InstanceNorm3d(affine=True, track_running_stats=False)
    - LeakyReLU(negative_slope=0.01)
    - output logits: ET, TC, WT

    When deep supervision is enabled, outputs are ordered from highest to lowest
    resolution. No sigmoid is applied inside the model.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: Sequence[int] = (32, 64, 128, 256, 320),
        *,
        convs_per_stage: int = 2,
        deep_supervision: bool = True,
        norm_eps: float = 1e-5,
        negative_slope: float = 1e-2,
    ) -> None:
        super().__init__()
        if len(features) < 2:
            raise ValueError("features must define at least two stages")
        if any(channel <= 0 for channel in features):
            raise ValueError("all feature counts must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.features = tuple(int(channel) for channel in features)
        self.deep_supervision = bool(deep_supervision)
        self.required_divisibility = 2 ** (len(self.features) - 1)

        encoder: list[nn.Module] = []
        previous_channels = self.in_channels
        for stage_index, stage_channels in enumerate(self.features):
            encoder.append(
                EncoderStage(
                    previous_channels,
                    stage_channels,
                    stride=1 if stage_index == 0 else 2,
                    num_convs=convs_per_stage,
                    norm_eps=norm_eps,
                    negative_slope=negative_slope,
                )
            )
            previous_channels = stage_channels
        self.encoder = nn.ModuleList(encoder)

        decoder: list[nn.Module] = []
        segmentation_heads: list[nn.Module] = []
        current_channels = self.features[-1]
        for skip_channels in reversed(self.features[:-1]):
            decoder.append(
                DecoderStage(
                    current_channels,
                    skip_channels,
                    skip_channels,
                    num_convs=convs_per_stage,
                    norm_eps=norm_eps,
                    negative_slope=negative_slope,
                )
            )
            segmentation_heads.append(nn.Conv3d(skip_channels, self.out_channels, kernel_size=1, bias=True))
            current_channels = skip_channels
        self.decoder = nn.ModuleList(decoder)
        self.segmentation_heads = nn.ModuleList(segmentation_heads)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2, mode="fan_out", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm3d) and module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"expected input shape [N, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        invalid = [size for size in x.shape[2:] if size % self.required_divisibility != 0]
        if invalid:
            raise ValueError(
                f"spatial dimensions {tuple(x.shape[2:])} must be divisible by {self.required_divisibility}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        self._check_input(x)
        skips: list[torch.Tensor] = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)

        decoder_logits: list[torch.Tensor] = []
        x = skips[-1]
        for stage, head, skip in zip(self.decoder, self.segmentation_heads, reversed(skips[:-1])):
            x = stage(x, skip)
            decoder_logits.append(head(x))

        # Decoder traversal is low -> high resolution; losses and callers expect high -> low.
        decoder_logits.reverse()
        if self.deep_supervision and self.training:
            return decoder_logits
        return decoder_logits[0]

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)


def build_source_model(model_config: dict) -> PlainUNet3D:
    """Build the source network from the validated YAML model section."""

    return PlainUNet3D(
        in_channels=model_config.get("in_channels", 4),
        out_channels=model_config.get("out_channels", 3),
        features=model_config.get("features", [32, 64, 128, 256, 320]),
        convs_per_stage=model_config.get("convs_per_stage", 2),
        deep_supervision=model_config.get("deep_supervision", True),
        norm_eps=model_config.get("norm_eps", 1e-5),
        negative_slope=model_config.get("negative_slope", 1e-2),
    )
