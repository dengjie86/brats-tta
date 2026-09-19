from __future__ import annotations

import math
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
        flip_axes: Sequence[int] = (0, 1, 2),
        affine_probability: float = 0.0,
        affine_scale_range: tuple[float, float] = (1.0, 1.0),
        affine_degrees: float = 0.0,
        gamma_probability: float = 0.0,
        gamma_log_range: tuple[float, float] = (0.0, 0.0),
        intensity_scale_probability: float = 1.0,
        intensity_scale_range: tuple[float, float] = (0.9, 1.1),
        intensity_shift_probability: float = 1.0,
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
        self.flip_axes = tuple(int(axis) for axis in flip_axes)
        if len(set(self.flip_axes)) != len(self.flip_axes) or any(
            axis not in {0, 1, 2} for axis in self.flip_axes
        ):
            raise ValueError("flip_axes must contain unique spatial axes selected from 0, 1 and 2")
        for name, probability in (
            ("affine_probability", affine_probability),
            ("gamma_probability", gamma_probability),
            ("intensity_scale_probability", intensity_scale_probability),
            ("intensity_shift_probability", intensity_shift_probability),
            ("noise_probability", noise_probability),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        _validate_range("affine_scale_range", affine_scale_range, positive=True)
        _validate_range("gamma_log_range", gamma_log_range)
        _validate_range("intensity_scale_range", intensity_scale_range, positive=True)
        _validate_range("intensity_shift_range", intensity_shift_range)
        _validate_range("noise_std_range", noise_std_range, nonnegative=True)
        if affine_degrees < 0:
            raise ValueError("affine_degrees must be nonnegative")
        self.foreground_oversample = float(foreground_oversample)
        self.flip_probability = float(flip_probability)
        self.affine_probability = float(affine_probability)
        self.affine_scale_range = affine_scale_range
        self.affine_degrees = float(affine_degrees)
        self.gamma_probability = float(gamma_probability)
        self.gamma_log_range = gamma_log_range
        self.intensity_scale_probability = float(intensity_scale_probability)
        self.intensity_scale_range = intensity_scale_range
        self.intensity_shift_probability = float(intensity_shift_probability)
        self.intensity_shift_range = intensity_shift_range
        self.noise_probability = float(noise_probability)
        self.noise_std_range = noise_std_range

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = _pad_to_shape(image, target, self.patch_size)
        starts = self._sample_crop_start(target)
        slices = tuple(slice(start, start + size) for start, size in zip(starts, self.patch_size))
        image = image[(slice(None), *slices)].clone()
        target = target[((*slices,) if target.ndim == 3 else (slice(None), *slices))].clone()

        for spatial_dimension in self.flip_axes:
            if torch.rand(()) < self.flip_probability:
                image_dimension = spatial_dimension + 1
                target_dimension = spatial_dimension if target.ndim == 3 else spatial_dimension + 1
                image = torch.flip(image, dims=(image_dimension,))
                target = torch.flip(target, dims=(target_dimension,))
        if torch.rand(()) < self.affine_probability:
            image, target = self._random_affine(image, target)
        image = self._intensity_augmentation(image)
        return image.contiguous(), target.contiguous()

    def _sample_crop_start(self, target: torch.Tensor) -> tuple[int, int, int]:
        spatial_shape = target.shape[-3:]
        foreground_map = target > 0 if target.ndim == 3 else target[2] > 0
        use_foreground = torch.rand(()) < self.foreground_oversample and torch.any(foreground_map)
        if use_foreground:
            foreground = torch.nonzero(foreground_map, as_tuple=False)
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
            if torch.rand(()) < self.gamma_probability:
                gamma = math.exp(_sample_uniform(*self.gamma_log_range))
                values = channel_image[foreground]
                channel_image[foreground] = values.sign() * values.abs().pow(gamma)
            if torch.rand(()) < self.intensity_scale_probability:
                scale = _sample_uniform(*self.intensity_scale_range)
                channel_image[foreground] = channel_image[foreground] * scale
            if torch.rand(()) < self.intensity_shift_probability:
                shift = _sample_uniform(*self.intensity_shift_range)
                channel_image[foreground] = channel_image[foreground] + shift
            if torch.rand(()) < self.noise_probability:
                noise_std = _sample_uniform(*self.noise_std_range)
                noise = torch.randn_like(channel_image[foreground]) * noise_std
                channel_image[foreground] = channel_image[foreground] + noise
        return image

    def _random_affine(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scales = torch.tensor(
            [_sample_uniform(*self.affine_scale_range) for _ in range(3)],
            dtype=image.dtype,
            device=image.device,
        )
        angles = torch.tensor(
            [math.radians(_sample_uniform(-self.affine_degrees, self.affine_degrees)) for _ in range(3)],
            dtype=image.dtype,
            device=image.device,
        )
        forward = _rotation_matrix_3d(angles) @ torch.diag(scales)
        theta = torch.zeros((1, 3, 4), dtype=image.dtype, device=image.device)
        theta[0, :, :3] = torch.linalg.inv(forward)
        grid = F.affine_grid(theta, size=(1, *image.shape), align_corners=False)
        transformed_image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

        target_dtype = target.dtype
        target_input = target.unsqueeze(0).float()
        if target.ndim == 3:
            target_input = target_input.unsqueeze(0)
        transformed_target = F.grid_sample(
            target_input,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        if target.ndim == 3:
            transformed_target = transformed_target.squeeze(0)
        return transformed_image, transformed_target.to(dtype=target_dtype)


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
    padded_target = F.pad(target, padding)
    return F.pad(image, padding), padded_target


def _sample_uniform(lower: float, upper: float) -> float:
    return float((lower + (upper - lower) * torch.rand(())).item())


def _rotation_matrix_3d(angles: torch.Tensor) -> torch.Tensor:
    x, y, z = angles.unbind()
    one = torch.ones((), dtype=angles.dtype, device=angles.device)
    zero = torch.zeros((), dtype=angles.dtype, device=angles.device)
    rx = torch.stack(
        (
            torch.stack((one, zero, zero)),
            torch.stack((zero, torch.cos(x), -torch.sin(x))),
            torch.stack((zero, torch.sin(x), torch.cos(x))),
        )
    )
    ry = torch.stack(
        (
            torch.stack((torch.cos(y), zero, torch.sin(y))),
            torch.stack((zero, one, zero)),
            torch.stack((-torch.sin(y), zero, torch.cos(y))),
        )
    )
    rz = torch.stack(
        (
            torch.stack((torch.cos(z), -torch.sin(z), zero)),
            torch.stack((torch.sin(z), torch.cos(z), zero)),
            torch.stack((zero, zero, one)),
        )
    )
    return rz @ ry @ rx


def _validate_range(
    name: str,
    values: tuple[float, float],
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> None:
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"{name} must contain an ordered lower and upper bound")
    if positive and values[0] <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and values[0] < 0:
        raise ValueError(f"{name} must be nonnegative")
