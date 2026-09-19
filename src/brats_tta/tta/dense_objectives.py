"""Label-free dense-prediction objectives for sigmoid segmentation TTA.

The source model predicts three overlapping Bernoulli regions.  Treating every
voxel equally makes the test-time objective overwhelmingly represent confident
background.  The reductions here make the spatial support and foreground /
background weighting explicit so experiments cannot silently change them.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

DenseReduction = Literal["all", "brain", "foreground_background_balanced"]
ConfidenceReduction = Literal["all", "brain", "foreground_background_balanced", "brain_uncertain"]


def binary_prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Bernoulli entropy for the independent ET/TC/WT sigmoid outputs."""

    probabilities = torch.sigmoid(logits.float()).clamp(1e-7, 1.0 - 1e-7)
    return -(probabilities * probabilities.log() + (1.0 - probabilities) * (1.0 - probabilities).log())


def image_brain_mask(images: torch.Tensor) -> torch.Tensor:
    """Return an image-derived [B,1,D,H,W] mask; padded/background voxels are zero."""

    if images.ndim != 5:
        raise ValueError(f"expected [B,C,D,H,W] images, got {tuple(images.shape)}")
    return images.detach().abs().sum(dim=1, keepdim=True) > 0


def _expanded_brain_mask(images: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    mask = image_brain_mask(images)
    if values.ndim != 5 or values.shape[0] != mask.shape[0] or values.shape[2:] != mask.shape[2:]:
        raise ValueError("images and dense values must share batch and spatial dimensions")
    return mask.expand(-1, values.shape[1], -1, -1, -1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if mask.any() else values.mean()


def _foreground_background_balanced_mean(
    values: torch.Tensor,
    probabilities: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Give every present foreground/background group equal weight per region."""

    if values.shape != probabilities.shape or values.shape != valid.shape:
        raise ValueError("values, probabilities, and valid mask must have identical shapes")
    detached_probabilities = probabilities.detach()
    group_means: list[torch.Tensor] = []
    for batch_index in range(values.shape[0]):
        for channel_index in range(values.shape[1]):
            channel_valid = valid[batch_index, channel_index]
            for predicted_foreground in (True, False):
                group = channel_valid & (
                    detached_probabilities[batch_index, channel_index] >= 0.5
                    if predicted_foreground
                    else detached_probabilities[batch_index, channel_index] < 0.5
                )
                if group.any():
                    group_means.append(values[batch_index, channel_index][group].mean())
    return torch.stack(group_means).mean() if group_means else values.mean()


def dense_entropy_loss(
    logits: torch.Tensor,
    images: torch.Tensor,
    *,
    reduction: DenseReduction = "all",
    normalize: bool = True,
) -> torch.Tensor:
    """Reduce independent Bernoulli entropy without target labels."""

    entropy = binary_prediction_entropy(logits)
    if reduction == "all":
        loss = entropy.mean()
    else:
        valid = _expanded_brain_mask(images, entropy)
        if reduction == "brain":
            loss = _masked_mean(entropy, valid)
        elif reduction == "foreground_background_balanced":
            loss = _foreground_background_balanced_mean(entropy, torch.sigmoid(logits.float()), valid)
        else:
            raise ValueError(f"unknown dense entropy reduction: {reduction}")
    return loss / math.log(2.0) if normalize else loss


def dense_teacher_consistency_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    images: torch.Tensor,
    *,
    reduction: DenseReduction = "all",
    normalize: bool = True,
) -> torch.Tensor:
    """Bernoulli teacher/student consistency with an explicit dense reduction."""

    if student_logits.shape != teacher_probabilities.shape:
        raise ValueError("student logits and teacher probabilities must have identical shapes")
    values = F.binary_cross_entropy_with_logits(
        student_logits.float(), teacher_probabilities.detach().float(), reduction="none"
    )
    if reduction == "all":
        loss = values.mean()
    else:
        valid = _expanded_brain_mask(images, values)
        if reduction == "brain":
            loss = _masked_mean(values, valid)
        elif reduction == "foreground_background_balanced":
            loss = _foreground_background_balanced_mean(values, teacher_probabilities.float(), valid)
        else:
            raise ValueError(f"unknown consistency reduction: {reduction}")
    return loss / math.log(2.0) if normalize else loss


def dense_prediction_confidence(
    logits: torch.Tensor,
    images: torch.Tensor,
    *,
    reduction: ConfidenceReduction = "all",
    uncertain_fraction: float = 0.05,
) -> torch.Tensor:
    """Measure confidence without allowing dense background to hide uncertain voxels."""

    if not 0 < uncertain_fraction <= 1:
        raise ValueError("uncertain_fraction must be in (0, 1]")
    probabilities = torch.sigmoid(logits.float())
    confidence = torch.maximum(probabilities, 1.0 - probabilities)
    if reduction == "all":
        return confidence.mean()
    valid = _expanded_brain_mask(images, confidence)
    if reduction == "brain":
        return _masked_mean(confidence, valid)
    if reduction == "foreground_background_balanced":
        return _foreground_background_balanced_mean(confidence, probabilities, valid)
    if reduction != "brain_uncertain":
        raise ValueError(f"unknown dense confidence reduction: {reduction}")

    # The least-confident brain voxels determine whether augmentation averaging
    # is needed.  This is the dense analogue of CoTTA's per-sample confidence
    # gate; a global mean is almost entirely a background-confidence measure.
    selected: list[torch.Tensor] = []
    for batch_index in range(confidence.shape[0]):
        for channel_index in range(confidence.shape[1]):
            values = confidence[batch_index, channel_index][valid[batch_index, channel_index]]
            if values.numel() == 0:
                continue
            count = max(1, math.ceil(values.numel() * uncertain_fraction))
            selected.append(values.topk(count, largest=False).values.mean())
    return torch.stack(selected).mean() if selected else confidence.mean()
