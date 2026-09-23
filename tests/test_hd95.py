from __future__ import annotations

import numpy as np
import pytest
import torch
from surface_distance import metrics as surface_distance_metrics

from brats_tta.metrics.segmentation import compute_region_metrics, hd95_binary


def test_hd95_uses_physical_spacing() -> None:
    prediction = np.zeros((5, 5, 5), dtype=bool)
    target = np.zeros_like(prediction)
    prediction[1, 2, 2] = True
    target[3, 2, 2] = True

    assert hd95_binary(prediction, target, spacing=(2.0, 1.0, 1.0)) == pytest.approx(4.0)


def test_hd95_empty_mask_convention() -> None:
    empty = np.zeros((3, 3, 3), dtype=bool)
    nonempty = empty.copy()
    nonempty[1, 1, 1] = True

    assert hd95_binary(empty, empty, spacing=(1.0, 1.0, 1.0)) == 0.0
    assert hd95_binary(empty, nonempty, spacing=(1.0, 1.0, 1.0)) == 374.0


def test_hd95_matches_area_weighted_implementation() -> None:
    prediction = np.zeros((7, 8, 9), dtype=bool)
    target = np.zeros_like(prediction)
    prediction[1:5, 2:7, 2:6] = True
    target[2:6, 1:6, 3:8] = True
    spacing = (1.5, 0.8, 2.0)

    distances = surface_distance_metrics.compute_surface_distances(
        target,
        prediction,
        spacing_mm=spacing,
    )
    expected = surface_distance_metrics.compute_robust_hausdorff(distances, 95)

    assert hd95_binary(prediction, target, spacing=spacing) == pytest.approx(expected)


def test_region_metrics_can_include_hd95() -> None:
    logits = torch.full((1, 4, 1, 1, 4), -10.0)
    for voxel, class_index in enumerate((0, 1, 2, 3)):
        logits[0, class_index, 0, 0, voxel] = 10.0
    target_regions = torch.tensor(
        [[[[[0, 0, 0, 1]]], [[[0, 1, 0, 1]]], [[[0, 1, 1, 1]]]]],
        dtype=torch.float32,
    )

    metrics = compute_region_metrics(
        logits,
        target_regions,
        output_mode="classes_softmax",
        label_schema="brats_modern",
        spacing=(1.0, 1.0, 1.0),
        include_hd95=True,
    )

    assert metrics["dice_mean"] == pytest.approx(1.0)
    assert metrics["hd95_ET"] == 0.0
    assert metrics["hd95_TC"] == 0.0
    assert metrics["hd95_WT"] == 0.0
    assert metrics["hd95_mean"] == 0.0
