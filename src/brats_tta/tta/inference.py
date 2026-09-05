from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
from torch import nn

from brats_tta.engine.inference import (
    _gaussian_importance_map,
    _pad_for_sliding_window,
    _scan_starts,
)
from brats_tta.tta.tent import TentAdapter


def sliding_window_tent_logits(
    model: nn.Module,
    adapter: TentAdapter,
    image: torch.Tensor,
    *,
    patch_size: Sequence[int],
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    gaussian_weighting: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Run online patch-by-patch Tent and stitch pre-update patch predictions."""

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
    entropies: list[float] = []

    for batch_start in range(0, len(locations), sw_batch_size):
        batch_locations = locations[batch_start : batch_start + sw_batch_size]
        patches = torch.cat(
            [
                image[:, :, d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]]
                for d, h, w in batch_locations
            ],
            dim=0,
        )
        result = adapter.predict_and_adapt(patches)
        batch_logits = result.logits
        entropies.append(result.entropy)
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
        with torch.no_grad():
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
    return logits, {
        "adaptation_updates": len(entropies),
        "adaptation_entropy": float(sum(entropies) / len(entropies)),
    }
