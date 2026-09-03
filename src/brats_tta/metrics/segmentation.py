from __future__ import annotations

import numpy as np
import torch

REGION_NAMES = ("ET", "TC", "WT")


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


def compute_region_metrics(
    logits_or_probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> dict[str, float]:
    probabilities = (
        torch.sigmoid(logits_or_probabilities.float()) if from_logits else logits_or_probabilities.float()
    )
    scores = dice_per_region(probabilities, target, threshold=threshold)
    metrics: dict[str, float] = {}
    for region_index, region_name in enumerate(REGION_NAMES):
        metrics[f"dice_{region_name}"] = float(scores[:, region_index].mean().item())
    metrics["dice_mean"] = float(scores.mean().item())
    metrics["hierarchy_violation"] = float(hierarchy_violation_rate(probabilities, threshold).mean().item())
    return metrics


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}
