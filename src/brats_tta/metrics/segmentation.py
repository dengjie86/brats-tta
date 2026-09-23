from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

try:
    from surface_distance import metrics as surface_distance_metrics
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal training images
    surface_distance_metrics = None

from brats_tta.data.preprocessing import classes_to_regions

REGION_NAMES = ("ET", "TC", "WT")
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)
_LESION_DILATION_STRUCTURE = ndimage.generate_binary_structure(3, 2)


def dice_per_region(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    empty_value: float = 1.0,
) -> torch.Tensor:
    if probabilities.shape != target.shape:
        raise ValueError(f"prediction {probabilities.shape} and target {target.shape} must match")
    prediction = probabilities >= threshold
    target_bool = target >= 0.5
    reduce_dimensions = (2, 3, 4)
    intersection = (prediction & target_bool).sum(dim=reduce_dimensions).float()
    denominator = (
        prediction.sum(dim=reduce_dimensions).float() + target_bool.sum(dim=reduce_dimensions).float()
    )
    fallback = torch.full_like(denominator, float(empty_value))
    return torch.where(denominator > 0, 2.0 * intersection / denominator, fallback)


def hierarchy_violation_rate(probabilities: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    prediction = probabilities >= threshold
    et, tc, wt = prediction.unbind(dim=1)
    violations = (et & ~tc) | (tc & ~wt)
    return violations.flatten(1).float().mean(dim=1)


def hd95_binary(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    empty_penalty: float = 374.0,
) -> float:
    """Compute the area-weighted symmetric robust Hausdorff distance at 95%."""

    if surface_distance_metrics is None:
        raise RuntimeError(
            "HD95 requires the optional 'surface-distance' package; install it before requesting "
            "include_hd95=True"
        )

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("HD95 masks must be matching 3D arrays")
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if spacing_array.shape != (3,) or not np.isfinite(spacing_array).all() or np.any(spacing_array <= 0):
        raise ValueError("spacing must contain three positive finite values")
    if not np.isfinite(empty_penalty) or empty_penalty < 0:
        raise ValueError("empty_penalty must be a non-negative finite value")

    prediction_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())
    if not prediction_nonempty and not target_nonempty:
        return 0.0
    if prediction_nonempty != target_nonempty:
        return float(empty_penalty)

    surface_distances = surface_distance_metrics.compute_surface_distances(
        target,
        prediction,
        spacing_mm=spacing_array,
    )
    return float(surface_distance_metrics.compute_robust_hausdorff(surface_distances, 95))


def hd95_per_region(
    prediction_regions: torch.Tensor | np.ndarray,
    target_regions: torch.Tensor | np.ndarray,
    *,
    spacing: tuple[float, float, float],
    threshold: float = 0.5,
    empty_penalty: float = 374.0,
) -> np.ndarray:
    prediction_array = np.asarray(
        prediction_regions.detach().cpu().numpy()
        if isinstance(prediction_regions, torch.Tensor)
        else prediction_regions
    )
    target_array = np.asarray(
        target_regions.detach().cpu().numpy()
        if isinstance(target_regions, torch.Tensor)
        else target_regions
    )
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"prediction {prediction_array.shape} and target {target_array.shape} must match"
        )
    if prediction_array.ndim != 5 or prediction_array.shape[1] != len(REGION_NAMES):
        raise ValueError("region arrays must have shape [B,3,D,H,W]")

    scores = np.empty((prediction_array.shape[0], len(REGION_NAMES)), dtype=np.float64)
    for batch_index in range(prediction_array.shape[0]):
        for region_index in range(len(REGION_NAMES)):
            scores[batch_index, region_index] = hd95_binary(
                prediction_array[batch_index, region_index] >= threshold,
                target_array[batch_index, region_index] >= 0.5,
                spacing=spacing,
                empty_penalty=empty_penalty,
            )
    return scores


def lesion_wise_dice_binary(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    dilation_factor: int = 3,
    volume_threshold_mm3: float = 50.0,
) -> dict[str, float | int]:
    """Compute the BraTS 2023 lesion-wise Dice and lesion detection counts."""

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("lesion-wise Dice masks must be matching 3D arrays")
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if spacing_array.shape != (3,) or not np.isfinite(spacing_array).all():
        raise ValueError("spacing must contain three finite values")
    if np.any(spacing_array <= 0):
        raise ValueError("spacing values must be positive")
    if dilation_factor < 0:
        raise ValueError("dilation_factor must be non-negative")
    if not np.isfinite(volume_threshold_mm3) or volume_threshold_mm3 < 0:
        raise ValueError("volume_threshold_mm3 must be non-negative and finite")

    target_labels, _ = ndimage.label(target, structure=_CONNECTIVITY_26)
    prediction_labels, prediction_count = ndimage.label(
        prediction,
        structure=_CONNECTIVITY_26,
    )
    dilated_target = ndimage.binary_dilation(
        target,
        structure=_LESION_DILATION_STRUCTURE,
        iterations=dilation_factor,
    )
    dilated_labels, dilated_count = ndimage.label(
        dilated_target,
        structure=_CONNECTIVITY_26,
    )

    # BraTS merges original GT components whose dilation extents connect.
    combined_target_labels = np.zeros_like(target_labels)
    for dilated_id in range(1, dilated_count + 1):
        original_ids = np.unique(target_labels[dilated_labels == dilated_id])
        original_ids = original_ids[original_ids != 0]
        if original_ids.size:
            combined_target_labels[np.isin(target_labels, original_ids)] = dilated_id

    voxel_volume_mm3 = float(np.prod(spacing_array))
    matched_prediction_ids: set[int] = set()
    dice_sum = 0.0
    evaluated_target_count = 0
    true_positive_count = 0
    false_negative_count = 0

    for target_id in range(1, dilated_count + 1):
        target_lesion = combined_target_labels == target_id
        if not target_lesion.any():
            continue
        target_roi = ndimage.binary_dilation(
            target_lesion,
            structure=_LESION_DILATION_STRUCTURE,
            iterations=dilation_factor,
        )
        intersecting_ids = np.unique(prediction_labels[target_roi])
        intersecting_ids = intersecting_ids[intersecting_ids != 0]
        matched_prediction_ids.update(int(value) for value in intersecting_ids)

        target_volume_mm3 = float(target_lesion.sum()) * voxel_volume_mm3
        if target_volume_mm3 <= volume_threshold_mm3:
            continue

        evaluated_target_count += 1
        if intersecting_ids.size == 0:
            false_negative_count += 1
            continue

        true_positive_count += 1
        matched_prediction = np.isin(prediction_labels, intersecting_ids)
        intersection = np.logical_and(matched_prediction, target_lesion).sum()
        denominator = matched_prediction.sum() + target_lesion.sum()
        dice_sum += 2.0 * float(intersection) / float(denominator)

    false_positive_count = prediction_count - len(matched_prediction_ids)
    denominator = evaluated_target_count + false_positive_count
    lesion_dice = 1.0 if denominator == 0 else dice_sum / denominator
    return {
        "dice": float(lesion_dice),
        "tp": int(true_positive_count),
        "fp": int(false_positive_count),
        "fn": int(false_negative_count),
    }


def lesion_wise_dice_per_region(
    prediction_regions: torch.Tensor | np.ndarray,
    target_regions: torch.Tensor | np.ndarray,
    *,
    spacing: tuple[float, float, float],
    threshold: float = 0.5,
    dilation_factor: int = 3,
    volume_threshold_mm3: float = 50.0,
) -> list[list[dict[str, float | int]]]:
    prediction_array = np.asarray(
        prediction_regions.detach().cpu().numpy()
        if isinstance(prediction_regions, torch.Tensor)
        else prediction_regions
    )
    target_array = np.asarray(
        target_regions.detach().cpu().numpy()
        if isinstance(target_regions, torch.Tensor)
        else target_regions
    )
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"prediction {prediction_array.shape} and target {target_array.shape} must match"
        )
    if prediction_array.ndim != 5 or prediction_array.shape[1] != len(REGION_NAMES):
        raise ValueError("region arrays must have shape [B,3,D,H,W]")

    return [
        [
            lesion_wise_dice_binary(
                prediction_array[batch_index, region_index] >= threshold,
                target_array[batch_index, region_index] >= 0.5,
                spacing=spacing,
                dilation_factor=dilation_factor,
                volume_threshold_mm3=volume_threshold_mm3,
            )
            for region_index in range(len(REGION_NAMES))
        ]
        for batch_index in range(prediction_array.shape[0])
    ]


def compute_region_metrics(
    logits_or_probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
    output_mode: str = "regions_sigmoid",
    label_schema: str = "brats_modern",
    spacing: tuple[float, float, float] | None = None,
    include_hd95: bool = False,
    hd95_empty_penalty: float = 374.0,
    include_lesion_wise: bool = False,
    lesion_dilation_factor: int = 3,
    lesion_volume_threshold_mm3: float = 50.0,
) -> dict[str, float]:
    if output_mode == "regions_sigmoid":
        probabilities = (
            torch.sigmoid(logits_or_probabilities.float()) if from_logits else logits_or_probabilities.float()
        )
        target_regions = target
        violation = hierarchy_violation_rate(probabilities, threshold).mean()
    elif output_mode == "classes_softmax":
        if logits_or_probabilities.ndim != 5 or logits_or_probabilities.shape[1] != 4:
            raise ValueError("classes_softmax metrics require [B,4,D,H,W] predictions")
        class_predictions = torch.argmax(logits_or_probabilities, dim=1)
        prediction_regions = torch.from_numpy(
            np.stack(
                [
                    classes_to_regions(case.detach().cpu().numpy(), label_schema)
                    for case in class_predictions
                ]
            )
        ).to(logits_or_probabilities.device, dtype=torch.float32)
        expected_spatial = tuple(logits_or_probabilities.shape[2:])
        if target.ndim == 5 and target.shape[1] == 3:
            if (
                target.shape[0] != logits_or_probabilities.shape[0]
                or tuple(target.shape[2:]) != expected_spatial
            ):
                raise ValueError("region target must be [B,3,D,H,W] matching prediction spatial shape")
            # Target domains can use a different mutually-exclusive label schema
            # (PED has an additional calcification class).  Comparing the
            # common ET/TC/WT regions avoids a lossy target-class remapping.
            target_regions = target.to(logits_or_probabilities.device, dtype=torch.float32)
        elif target.ndim == 4:
            if tuple(target.shape) != (logits_or_probabilities.shape[0], *expected_spatial):
                raise ValueError("class target must be [B,D,H,W] matching prediction spatial shape")
            target_regions = torch.from_numpy(
                np.stack(
                    [
                        classes_to_regions(case.detach().cpu().numpy(), label_schema)
                        for case in target.long()
                    ]
                )
            ).to(logits_or_probabilities.device, dtype=torch.float32)
        else:
            raise ValueError("classes_softmax target must be a class map or three ET/TC/WT regions")
        probabilities = prediction_regions
        violation = torch.zeros((), device=logits_or_probabilities.device)
    else:
        raise ValueError(f"unknown output_mode {output_mode!r}")
    scores = dice_per_region(probabilities, target_regions, threshold=threshold)
    metrics: dict[str, float] = {}
    for region_index, region_name in enumerate(REGION_NAMES):
        metrics[f"dice_{region_name}"] = float(scores[:, region_index].mean().item())
    metrics["dice_mean"] = float(scores.mean().item())
    metrics["hierarchy_violation"] = float(violation.item())
    if include_hd95:
        if spacing is None:
            raise ValueError("spacing is required when include_hd95=True")
        hd95_scores = hd95_per_region(
            probabilities,
            target_regions,
            spacing=spacing,
            threshold=threshold,
            empty_penalty=hd95_empty_penalty,
        )
        for region_index, region_name in enumerate(REGION_NAMES):
            metrics[f"hd95_{region_name}"] = float(hd95_scores[:, region_index].mean())
        metrics["hd95_mean"] = float(hd95_scores.mean())
    if include_lesion_wise:
        if spacing is None:
            raise ValueError("spacing is required when include_lesion_wise=True")
        lesion_results = lesion_wise_dice_per_region(
            probabilities,
            target_regions,
            spacing=spacing,
            threshold=threshold,
            dilation_factor=lesion_dilation_factor,
            volume_threshold_mm3=lesion_volume_threshold_mm3,
        )
        for region_index, region_name in enumerate(REGION_NAMES):
            region_results = [case[region_index] for case in lesion_results]
            metrics[f"lesionwise_dice_{region_name}"] = float(
                np.mean([float(result["dice"]) for result in region_results])
            )
            for count_name in ("tp", "fp", "fn"):
                metrics[f"lesionwise_{count_name}_{region_name}"] = float(
                    np.mean([int(result[count_name]) for result in region_results])
                )
        metrics["lesionwise_dice_mean"] = float(
            np.mean(
                [
                    float(result["dice"])
                    for case_results in lesion_results
                    for result in case_results
                ]
            )
        )
    return metrics


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}
