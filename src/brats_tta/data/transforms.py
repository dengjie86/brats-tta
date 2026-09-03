from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


class SourcePatchTransform:
    def __init__(
        self,
        patch_size: Sequence[int],
        *,
        foreground_oversample: float = 0.33,
        flip_probability: float = 0.5,
        intensity_scale_range: tuple[float, float] = (0.9, 1.1),
        intensity_shift_range: tuple[float, float] = (-0.1, 0.1),
        noise_probability: float = 0.15,
        noise_std_range: tuple[float, float] = (0.0, 0.1),
    ) -> None:
        self.patch_size = tuple(int(size) for size in patch_size)
        if len(self.patch_size) != 3 or any(size <= 0 for size in self.patch_size):
            raise ValueError("patch_size must contain three positive values")
        if not 0 <= foreground_oversample <= 1:
            raise ValueError("foreground_oversample must be in [0, 1]")
        if not 0 <= flip_probability <= 1:
            raise ValueError("flip_probability must be in [0, 1]")
        if not 0 <= noise_probability <= 1:
            raise ValueError("noise_probability must be in [0, 1]")
        self.foreground_oversample = float(foreground_oversample)
        self.flip_probability = float(flip_probability)
        self.intensity_scale_range = intensity_scale_range
        self.intensity_shift_range = intensity_shift_range
        self.noise_probability = float(noise_probability)
        self.noise_std_range = noise_std_range

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = _pad_to_shape(image, target, self.patch_size)
        starts = self._sample_crop_start(target)
        slices = tuple(slice(start, start + size) for start, size in zip(starts, self.patch_size))
        image = image[(slice(None), *slices)].clone()
        target = target[(slice(None), *slices)].clone()

        for spatial_dimension in range(3):
            if torch.rand(()) < self.flip_probability:
                tensor_dimension = spatial_dimension + 1
                image = torch.flip(image, dims=(tensor_dimension,))
                target = torch.flip(target, dims=(tensor_dimension,))
        image = self._intensity_augmentation(image)
        return image.contiguous(), target.contiguous()

    def _sample_crop_start(self, target: torch.Tensor) -> tuple[int, int, int]:
        spatial_shape = target.shape[1:]
        use_foreground = torch.rand(()) < self.foreground_oversample and torch.any(target[2] > 0)
        if use_foreground:
            foreground = torch.nonzero(target[2] > 0, as_tuple=False)
            center = foreground[torch.randint(len(foreground), size=())]
            starts = []
            for center_coordinate, image_size, patch_size in zip(
                center.tolist(), spatial_shape, self.patch_size
            ):
                maximum_start = image_size - patch_size
                starts.append(max(0, min(center_coordinate - patch_size // 2, maximum_start)))
            return tuple(starts)

        return tuple(
            int(torch.randint(image_size - patch_size + 1, size=()).item()) if image_size > patch_size else 0
            for image_size, patch_size in zip(spatial_shape, self.patch_size)
        )

    def _intensity_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        for channel in range(image.shape[0]):
            channel_image = image[channel]
            foreground = channel_image != 0
            if not torch.any(foreground):
                continue
            scale = _sample_uniform(*self.intensity_scale_range)
            shift = _sample_uniform(*self.intensity_shift_range)
            channel_image[foreground] = channel_image[foreground] * scale + shift
            if torch.rand(()) < self.noise_probability:
                noise_std = _sample_uniform(*self.noise_std_range)
                noise = torch.randn_like(channel_image[foreground]) * noise_std
                channel_image[foreground] = channel_image[foreground] + noise
        return image


def _pad_to_shape(
    image: torch.Tensor,
    target: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    padding_per_dimension: list[tuple[int, int]] = []
    for current, required in zip(image.shape[1:], patch_size):
        total = max(0, required - current)
        before = total // 2
        padding_per_dimension.append((before, total - before))
    if not any(before or after for before, after in padding_per_dimension):
        return image, target
    padding = tuple(value for pair in reversed(padding_per_dimension) for value in pair)
    return F.pad(image, padding), F.pad(target, padding)


def _sample_uniform(lower: float, upper: float) -> float:
    return float((lower + (upper - lower) * torch.rand(())).item())
