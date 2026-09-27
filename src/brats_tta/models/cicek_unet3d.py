from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

CicekNorm = Literal["none", "batch3d"]
OutputMode = Literal["regions_sigmoid", "classes_softmax"]


def _make_norm(
    norm: CicekNorm,
    channels: int,
    *,
    eps: float,
    momentum: float,
    track_running_stats: bool,
) -> nn.Module | None:
    if norm == "none":
        return None
    if norm == "batch3d":
        return nn.BatchNorm3d(
            channels,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=track_running_stats,
        )
    raise ValueError(f"unsupported Cicek 3D U-Net normalization: {norm}")


class ConvNormReLU3D(nn.Module):
    """One 3D convolution followed by optional BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: CicekNorm,
        norm_eps: float,
        norm_momentum: float,
        track_running_stats: bool,
        bias: bool,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.norm = _make_norm(
            norm,
            out_channels,
            eps=norm_eps,
            momentum=norm_momentum,
            track_running_stats=track_running_stats,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.relu(x)


class CicekEncoderStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        *,
        norm: CicekNorm,
        norm_eps: float,
        norm_momentum: float,
        track_running_stats: bool,
        first_conv_bias: bool,
    ) -> None:
        super().__init__()
        convolution_bias = norm == "none"
        self.conv1 = ConvNormReLU3D(
            in_channels,
            middle_channels,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            track_running_stats=track_running_stats,
            bias=first_conv_bias or convolution_bias,
        )
        self.conv2 = ConvNormReLU3D(
            middle_channels,
            out_channels,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            track_running_stats=track_running_stats,
            bias=convolution_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class CicekDecoderStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        *,
        norm: CicekNorm,
        norm_eps: float,
        norm_momentum: float,
        track_running_stats: bool,
    ) -> None:
        super().__init__()
        convolution_bias = norm == "none"
        # The released 3D U-Net keeps the incoming channel count through the
        # transposed convolution, then reduces it after skip concatenation.
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
            bias=convolution_bias,
        )
        self.up_norm = _make_norm(
            norm,
            in_channels,
            eps=norm_eps,
            momentum=norm_momentum,
            track_running_stats=track_running_stats,
        )
        self.up_relu = nn.ReLU(inplace=True)
        self.conv1 = ConvNormReLU3D(
            in_channels + skip_channels,
            skip_channels,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            track_running_stats=track_running_stats,
            bias=convolution_bias,
        )
        self.conv2 = ConvNormReLU3D(
            skip_channels,
            skip_channels,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            track_running_stats=track_running_stats,
            bias=convolution_bias,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.up_norm is not None:
            x = self.up_norm(x)
        x = self.up_relu(x)
        if x.shape[2:] != skip.shape[2:]:
            raise RuntimeError(
                "decoder and skip shapes do not match; input dimensions must be divisible "
                f"by the network downsampling factor, got {x.shape[2:]} and {skip.shape[2:]}"
            )
        return self.conv2(self.conv1(torch.cat((skip, x), dim=1)))


class CicekUNet3D(nn.Module):
    """BraTS adaptation of the 2016 Cicek et al. 3D U-Net.

    The released architecture uses four resolution levels with encoder outputs
    ``64, 128, 256, 512``. Every convolution and transposed convolution is
    followed by BatchNorm and ReLU. This implementation retains that channel
    topology but uses same padding so a BraTS patch and its label stay aligned.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: Sequence[int] = (64, 128, 256, 512),
        *,
        output_mode: OutputMode | None = None,
        norm: CicekNorm = "batch3d",
        norm_eps: float = 1e-5,
        norm_momentum: float = 1e-3,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        if len(features) != 4:
            raise ValueError("CicekUNet3D requires four resolution levels")
        if any(int(channel) <= 0 or int(channel) % 2 for channel in features):
            raise ValueError("CicekUNet3D feature counts must be positive even integers")
        if any(int(right) != 2 * int(left) for left, right in zip(features, features[1:])):
            raise ValueError("CicekUNet3D feature counts must double at each level")
        if norm not in {"none", "batch3d"}:
            raise ValueError(f"unsupported Cicek 3D U-Net normalization: {norm}")
        if norm == "batch3d" and norm_momentum <= 0:
            raise ValueError("BatchNorm3d norm_momentum must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if output_mode is None:
            output_mode = "regions_sigmoid" if self.out_channels == 3 else "classes_softmax"
        if output_mode not in {"regions_sigmoid", "classes_softmax"}:
            raise ValueError(f"unsupported output_mode: {output_mode}")
        expected_channels = 3 if output_mode == "regions_sigmoid" else 4
        if self.out_channels != expected_channels:
            raise ValueError(
                f"{output_mode} requires out_channels={expected_channels}, got {self.out_channels}"
            )

        self.features = tuple(int(channel) for channel in features)
        self.output_mode: OutputMode = output_mode
        self.norm: CicekNorm = norm
        self.padding = "same"
        self.norm_eps = float(norm_eps)
        self.norm_momentum = float(norm_momentum)
        self.track_running_stats = bool(track_running_stats)
        self.deep_supervision = False
        self.required_divisibility = 2 ** (len(self.features) - 1)

        encoder: list[nn.Module] = []
        previous_channels = self.in_channels
        for stage_index, stage_channels in enumerate(self.features):
            encoder.append(
                CicekEncoderStage3D(
                    previous_channels,
                    stage_channels // 2,
                    stage_channels,
                    norm=norm,
                    norm_eps=norm_eps,
                    norm_momentum=norm_momentum,
                    track_running_stats=track_running_stats,
                    first_conv_bias=stage_index == 0 and norm == "batch3d",
                )
            )
            previous_channels = stage_channels
        self.encoder = nn.ModuleList(encoder)
        self.pool = nn.ModuleList(
            nn.MaxPool3d(kernel_size=2, stride=2) for _ in range(len(self.features) - 1)
        )

        decoder: list[nn.Module] = []
        current_channels = self.features[-1]
        for skip_channels in reversed(self.features[:-1]):
            decoder.append(
                CicekDecoderStage3D(
                    current_channels,
                    skip_channels,
                    norm=norm,
                    norm_eps=norm_eps,
                    norm_momentum=norm_momentum,
                    track_running_stats=track_running_stats,
                )
            )
            current_channels = skip_channels
        self.decoder = nn.ModuleList(decoder)
        self.head = nn.Conv3d(current_channels, self.out_channels, kernel_size=1, bias=True)

        self.apply(self._initialize_weights)
        first_bias = self.encoder[0].conv1.conv.bias
        if first_bias is not None:
            nn.init.constant_(first_bias, -0.1)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"expected input shape [N, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        if any(size % self.required_divisibility for size in x.shape[2:]):
            raise ValueError(
                f"spatial dimensions {tuple(x.shape[2:])} must be divisible by "
                f"{self.required_divisibility}"
            )

    def encode_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._check_input(x)
        features: list[torch.Tensor] = []
        for stage_index, stage in enumerate(self.encoder):
            x = stage(x)
            features.append(x)
            if stage_index < len(self.pool):
                x = self.pool[stage_index](x)
        return features

    def decode_features(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.encoder):
            raise ValueError(f"expected {len(self.encoder)} encoder features, got {len(features)}")
        x = features[-1]
        for stage, skip in zip(self.decoder, reversed(features[:-1])):
            x = stage(x, skip)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_features(self.encode_features(x))

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)


def build_cicek_unet3d(model_config: dict) -> CicekUNet3D:
    return CicekUNet3D(
        in_channels=model_config.get("in_channels", 4),
        out_channels=model_config.get("out_channels", 4),
        features=model_config.get("features", [64, 128, 256, 512]),
        output_mode=model_config.get("output_mode"),
        norm=model_config.get("norm", "batch3d"),
        norm_eps=model_config.get("norm_eps", 1e-5),
        norm_momentum=model_config.get("norm_momentum", 1e-3),
        track_running_stats=model_config.get("track_running_stats", False),
    )
