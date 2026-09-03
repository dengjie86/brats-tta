from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from brats_tta.data.preprocessing import regions_to_labelmap


@torch.no_grad()
def sliding_window_logits(
    model: nn.Module,
    image: torch.Tensor,
    *,
    patch_size: Sequence[int],
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    gaussian_weighting: bool = True,
    amp: bool = True,
) -> torch.Tensor:
    """Predict a single 3D case and return logits with the original spatial shape."""

    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError(f"expected one image [1, C, D, H, W], got {tuple(image.shape)}")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if sw_batch_size < 1:
        raise ValueError("sw_batch_size must be positive")
    patch_size = tuple(int(size) for size in patch_size)
    if len(patch_size) != 3 or any(size <= 0 for size in patch_size):
        raise ValueError("patch_size must contain three positive values")
    required_divisibility = getattr(model, "required_divisibility", None)
    if required_divisibility is not None and any(size % required_divisibility for size in patch_size):
        raise ValueError(f"patch_size must be divisible by the model factor {required_divisibility}")
    original_shape = tuple(image.shape[2:])
    image, crop_slices = _pad_for_sliding_window(image, patch_size)
    padded_shape = tuple(image.shape[2:])
    starts = [_scan_starts(size, patch, overlap) for size, patch in zip(padded_shape, patch_size)]
    locations = list(itertools.product(*starts))

    importance = (
        _gaussian_importance_map(patch_size, image.device)
        if gaussian_weighting
        else torch.ones((1, 1, *patch_size), device=image.device, dtype=torch.float32)
    )
    output_accumulator: torch.Tensor | None = None
    weight_accumulator: torch.Tensor | None = None

    for batch_start in range(0, len(locations), sw_batch_size):
        batch_locations = locations[batch_start : batch_start + sw_batch_size]
        patches = torch.cat(
            [
                image[
                    :,
                    :,
                    d : d + patch_size[0],
                    h : h + patch_size[1],
                    w : w + patch_size[2],
                ]
                for d, h, w in batch_locations
            ],
            dim=0,
        )
        use_amp = bool(amp and patches.device.type == "cuda")
        with torch.autocast(device_type=patches.device.type, dtype=torch.float16, enabled=use_amp):
            batch_logits = model(patches)
            if isinstance(batch_logits, (tuple, list)):
                batch_logits = batch_logits[0]
        batch_logits = batch_logits.float()
        if batch_logits.ndim != 5 or tuple(batch_logits.shape[2:]) != patch_size:
            raise RuntimeError(
                f"model returned shape {tuple(batch_logits.shape)} for sliding-window patch {patch_size}"
            )

        if output_accumulator is None:
            output_accumulator = torch.zeros(
                (1, batch_logits.shape[1], *padded_shape),
                dtype=torch.float32,
                device=image.device,
            )
            weight_accumulator = torch.zeros(
                (1, 1, *padded_shape),
                dtype=torch.float32,
                device=image.device,
            )
        assert weight_accumulator is not None
        for patch_index, (d, h, w) in enumerate(batch_locations):
            spatial_slice = (
                slice(d, d + patch_size[0]),
                slice(h, h + patch_size[1]),
                slice(w, w + patch_size[2]),
            )
            output_accumulator[(slice(None), slice(None), *spatial_slice)] += (
                batch_logits[patch_index : patch_index + 1] * importance
            )
            weight_accumulator[(slice(None), slice(None), *spatial_slice)] += importance

    assert output_accumulator is not None and weight_accumulator is not None
    logits = output_accumulator / weight_accumulator.clamp_min(1e-7)
    logits = logits[(slice(None), slice(None), *crop_slices)]
    if tuple(logits.shape[2:]) != original_shape:
        raise RuntimeError(f"sliding-window crop returned {logits.shape[2:]}, expected {original_shape}")
    return logits


def save_brats_prediction(
    probabilities: torch.Tensor | np.ndarray,
    reference_path: str | Path,
    destination: str | Path,
    *,
    label_schema: str,
    threshold: float = 0.5,
    enforce_hierarchy: bool = True,
) -> None:
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()
    if probabilities.ndim == 5:
        probabilities = probabilities[0]
    labelmap = regions_to_labelmap(
        probabilities,
        label_schema,
        threshold=threshold,
        enforce_hierarchy=enforce_hierarchy,
    )
    reference = nib.load(str(reference_path))
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(labelmap, reference.affine, header), str(destination))


def _scan_starts(image_size: int, patch_size: int, overlap: float) -> list[int]:
    if image_size <= patch_size:
        return [0]
    step = max(1, int(round(patch_size * (1.0 - overlap))))
    starts = list(range(0, image_size - patch_size + 1, step))
    final_start = image_size - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _pad_for_sliding_window(
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[slice, slice, slice]]:
    padding_per_dimension: list[tuple[int, int]] = []
    crop_slices: list[slice] = []
    for image_size, required in zip(image.shape[2:], patch_size):
        total = max(0, required - image_size)
        before = total // 2
        after = total - before
        padding_per_dimension.append((before, after))
        crop_slices.append(slice(before, before + image_size))
    if any(before or after for before, after in padding_per_dimension):
        padding = tuple(value for pair in reversed(padding_per_dimension) for value in pair)
        image = F.pad(image, padding)
    return image, tuple(crop_slices)  # type: ignore[return-value]


def _gaussian_importance_map(patch_size: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    axes = [torch.linspace(-1.0, 1.0, steps=size, device=device) for size in patch_size]
    grid = torch.meshgrid(*axes, indexing="ij")
    squared_distance = sum(axis.square() for axis in grid)
    importance = torch.exp(-squared_distance / (2 * 0.5**2))
    importance = importance / importance.max()
    return importance.clamp_min(1e-3)[None, None].float()
