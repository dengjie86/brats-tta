from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

OriginalNorm = Literal["none", "batch3d"]
OutputMode = Literal["regions_sigmoid", "classes_softmax"]


def _make_norm(
    norm: OriginalNorm,
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
    raise ValueError(f"unsupported original U-Net normalization: {norm}")


class OriginalConvBlock3D(nn.Module):
    """Two same-padded 3D convolutions followed by ReLU activations.

    The ``batch3d`` option is an explicitly named compatibility control for
    TENT experiments. The topology control uses ``norm='none'`` and therefore
    has no normalization, dropout, residual path, or leaky activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: OriginalNorm,
        norm_eps: float,
        norm_momentum: float,
        track_running_stats: bool,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=norm == "none",
        )
        self.norm1 = _make_norm(
            norm,
            out_channels,
            eps=norm_eps,
            momentum=norm_momentum,
            track_running_stats=track_running_stats,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=norm == "none",
        )
        self.norm2 = _make_norm(
            norm,
            out_channels,
            eps=norm_eps,
            momentum=norm_momentum,
            track_running_stats=track_running_stats,
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        return self.relu2(x)


class OriginalUNet3D(nn.Module):
    """Original U-Net topology implemented with 3D operators.

    This is the standard encoder-decoder topology from the U-Net paper:
    repeated two-convolution blocks, max-pooling downsampling, transposed
    convolution upsampling, concatenated skip connections, and a 1x1 output
    head. Same padding is used so a native BraTS patch keeps its spatial size.

    This is a topology-faithful 3D adaptation, not a literal reproduction of
    the paper's 2D valid-convolution border-cropping geometry. ``norm='none'``
    is the plain U-Net control; ``norm='batch3d'`` is provided only as a paired
    normalization control for TENT ablations and does not add dropout,
    residual connections, deep supervision, or other blocks.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: Sequence[int] = (32, 64, 128, 256, 512),
        *,
        output_mode: OutputMode | None = None,
        norm: OriginalNorm = "none",
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if len(features) < 2:
            raise ValueError("features must define at least two resolution levels")
        if any(int(channel) <= 0 for channel in features):
            raise ValueError("all feature counts must be positive")
        if norm not in {"none", "batch3d"}:
            raise ValueError(f"unsupported original U-Net normalization: {norm}")
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
        self.norm: OriginalNorm = norm
        self.padding = "same"
        self.norm_eps = float(norm_eps)
        self.norm_momentum = float(norm_momentum)
        self.track_running_stats = bool(track_running_stats)
        self.deep_supervision = False
        self.required_divisibility = 2 ** (len(self.features) - 1)

        encoder: list[nn.Module] = []
        previous_channels = self.in_channels
        for stage_channels in self.features:
            encoder.append(
                OriginalConvBlock3D(
                    previous_channels,
                    stage_channels,
                    norm=norm,
                    norm_eps=norm_eps,
                    norm_momentum=norm_momentum,
                    track_running_stats=track_running_stats,
                )
            )
            previous_channels = stage_channels
        self.encoder = nn.ModuleList(encoder)
        self.pool = nn.ModuleList(
            nn.MaxPool3d(kernel_size=2, stride=2) for _ in range(len(self.features) - 1)
        )

        upsamplers: list[nn.Module] = []
        decoder: list[nn.Module] = []
        current_channels = self.features[-1]
        for skip_channels in reversed(self.features[:-1]):
            upsamplers.append(
                nn.ConvTranspose3d(
                    current_channels,
                    skip_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            decoder.append(
                OriginalConvBlock3D(
                    skip_channels * 2,
                    skip_channels,
                    norm=norm,
                    norm_eps=norm_eps,
                    norm_momentum=norm_momentum,
                    track_running_stats=track_running_stats,
                )
            )
            current_channels = skip_channels
        self.upsamplers = nn.ModuleList(upsamplers)
        self.decoder = nn.ModuleList(decoder)
        self.head = nn.Conv3d(current_channels, self.out_channels, kernel_size=1)

        self.apply(self._initialize_weights)

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
        invalid = [size for size in x.shape[2:] if size % self.required_divisibility]
        if invalid:
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
        for upsample, block, skip in zip(
            self.upsamplers,
            self.decoder,
            reversed(features[:-1]),
        ):
            x = upsample(x)
            if x.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    "decoder and skip shapes do not match; input dimensions must be divisible "
                    f"by {self.required_divisibility}, got {x.shape[2:]} and {skip.shape[2:]}"
                )
            x = block(torch.cat((skip, x), dim=1))
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_features(self.encode_features(x))

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)


def build_original_unet3d(model_config: dict) -> OriginalUNet3D:
    """Build the strict/original-topology model from a validated config."""

    return OriginalUNet3D(
        in_channels=model_config.get("in_channels", 4),
        out_channels=model_config.get("out_channels", 4),
        features=model_config.get("features", [32, 64, 128, 256, 512]),
        output_mode=model_config.get("output_mode"),
        norm=model_config.get("norm", "none"),
        norm_eps=model_config.get("norm_eps", 1e-5),
        norm_momentum=model_config.get("norm_momentum", 0.1),
        track_running_stats=model_config.get("track_running_stats", True),
    )
