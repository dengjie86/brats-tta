from __future__ import annotations

import numpy as np
import pytest
import torch

from brats_tta.metrics.segmentation import compute_region_metrics, lesion_wise_dice_binary


def _cube(shape: tuple[int, int, int], start: tuple[int, int, int], size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    slices = tuple(slice(offset, offset + size) for offset in start)
    mask[slices] = True
    return mask


def test_lesion_wise_dice_perfect_detection() -> None:
    target = _cube((16, 16, 16), (3, 3, 3), 4)

    result = lesion_wise_dice_binary(target, target, spacing=(1.0, 1.0, 1.0))

    assert result == {"dice": 1.0, "tp": 1, "fp": 0, "fn": 0}


def test_lesion_wise_dice_penalizes_false_positive_lesions() -> None:
    target = _cube((24, 24, 24), (3, 3, 3), 4)
    prediction = target | _cube((24, 24, 24), (16, 16, 16), 2)

    result = lesion_wise_dice_binary(prediction, target, spacing=(1.0, 1.0, 1.0))

    assert result["dice"] == pytest.approx(0.5)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_lesion_wise_dice_penalizes_missed_lesions() -> None:
    target = _cube((16, 16, 16), (3, 3, 3), 4)

    result = lesion_wise_dice_binary(
        np.zeros_like(target),
        target,
        spacing=(1.0, 1.0, 1.0),
    )

    assert result == {"dice": 0.0, "tp": 0, "fp": 0, "fn": 1}


def test_small_ground_truth_lesions_are_excluded_like_brats_2023() -> None:
    target = _cube((16, 16, 16), (3, 3, 3), 3)

    result = lesion_wise_dice_binary(target, target, spacing=(1.0, 1.0, 1.0))

    assert result == {"dice": 1.0, "tp": 0, "fp": 0, "fn": 0}


def test_volume_threshold_uses_physical_spacing() -> None:
    target = _cube((16, 16, 16), (3, 3, 3), 3)

    result = lesion_wise_dice_binary(target, target, spacing=(2.0, 1.0, 1.0))

    assert result == {"dice": 1.0, "tp": 1, "fp": 0, "fn": 0}


def test_compute_region_metrics_includes_lesion_wise_scores() -> None:
    logits = torch.full((1, 4, 8, 8, 8), -10.0)
    logits[:, 0] = 10.0
    logits[:, 0, 2:6, 2:6, 2:6] = -10.0
    logits[:, 3, 2:6, 2:6, 2:6] = 10.0
    target = torch.zeros((1, 3, 8, 8, 8), dtype=torch.float32)
    target[:, :, 2:6, 2:6, 2:6] = 1.0

    metrics = compute_region_metrics(
        logits,
        target,
        output_mode="classes_softmax",
        label_schema="brats_modern",
        spacing=(1.0, 1.0, 1.0),
        include_lesion_wise=True,
    )

    assert metrics["lesionwise_dice_ET"] == pytest.approx(1.0)
    assert metrics["lesionwise_dice_TC"] == pytest.approx(1.0)
    assert metrics["lesionwise_dice_WT"] == pytest.approx(1.0)
    assert metrics["lesionwise_dice_mean"] == pytest.approx(1.0)
    assert metrics["lesionwise_tp_ET"] == 1.0
