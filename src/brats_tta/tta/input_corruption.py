"""Deterministic label-free MRI perturbations for source-validation diagnostics."""

from __future__ import annotations

import torch

from brats_tta.tta.dense_objectives import image_brain_mask


def _masked_channel_zscore(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(images)
    for batch_index in range(images.shape[0]):
        spatial_mask = mask[batch_index, 0]
        for channel_index in range(images.shape[1]):
            values = images[batch_index, channel_index][spatial_mask]
            if values.numel() == 0:
                continue
            standard_deviation = values.std(unbiased=False).clamp_min(1e-6)
            result[batch_index, channel_index][spatial_mask] = (values - values.mean()) / standard_deviation
    return result


def apply_mri_mixed_corruption(images: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Apply nonlinear contrast, smooth bias field, and moderate Gaussian noise.

    The image-derived brain support is preserved and each modality is z-scored
    again, matching the deployment preprocessing contract.  This is a source
    validation stress test, not an attempt to synthesize a particular target.
    """

    if images.ndim != 5 or images.shape[1] != 4:
        raise ValueError("MRI corruption expects [B,4,D,H,W]")
    mask = image_brain_mask(images)
    working = images.float().clone()
    gammas = torch.tensor((1.25, 0.80, 1.40, 1.10), device=images.device).view(1, 4, 1, 1, 1)
    working = working.sign() * working.abs().clamp_max(12.0).pow(gammas)

    depth = torch.linspace(-1.0, 1.0, images.shape[-3], device=images.device).view(1, 1, -1, 1, 1)
    height = torch.linspace(-1.0, 1.0, images.shape[-2], device=images.device).view(1, 1, 1, -1, 1)
    width = torch.linspace(-1.0, 1.0, images.shape[-1], device=images.device).view(1, 1, 1, 1, -1)
    bias_field = (1.0 + 0.20 * depth - 0.15 * height + 0.10 * width).clamp(0.55, 1.45)
    working = working * bias_field

    generator = torch.Generator(device=images.device).manual_seed(int(seed))
    noise = torch.randn(images.shape, generator=generator, device=images.device, dtype=torch.float32)
    working = working + 0.10 * noise
    working = torch.where(mask, working, torch.zeros((), device=images.device))
    return _masked_channel_zscore(working, mask).to(images.dtype)
