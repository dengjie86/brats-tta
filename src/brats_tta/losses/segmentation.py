from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, batch_dice: bool = False) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.batch_dice = bool(batch_dice)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits.float())
        target = target.float()
        if probabilities.shape != target.shape:
            raise ValueError(f"prediction {probabilities.shape} and target {target.shape} must match")

        if self.batch_dice:
            reduce_dimensions = (0, 2, 3, 4)
        else:
            reduce_dimensions = (2, 3, 4)
        intersection = (probabilities * target).sum(dim=reduce_dimensions)
        denominator = probabilities.sum(dim=reduce_dimensions) + target.sum(dim=reduce_dimensions)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class HierarchyLoss(nn.Module):
    """Penalize ET > TC or TC > WT for ET/TC/WT region probabilities."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != 3:
            raise ValueError("HierarchyLoss expects channels ordered as ET, TC, WT")
        probabilities = torch.sigmoid(logits.float())
        et, tc, wt = probabilities.unbind(dim=1)
        return F.relu(et - tc).mean() + F.relu(tc - wt).mean()


class BraTSRegionLoss(nn.Module):
    """Dice + BCEWithLogits for overlapping ET, TC and WT targets."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        hierarchy_weight: float = 0.0,
        smooth: float = 1e-5,
        batch_dice: bool = False,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.dice = SoftDiceLoss(smooth=smooth, batch_dice=batch_dice)
        self.hierarchy = HierarchyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.dice_weight * self.dice(logits, target)
        loss = loss + self.bce_weight * F.binary_cross_entropy_with_logits(logits.float(), target.float())
        if self.hierarchy_weight > 0:
            loss = loss + self.hierarchy_weight * self.hierarchy(logits)
        return loss


class DeepSupervisionLoss(nn.Module):
    """Apply a region loss to high-to-low-resolution decoder outputs."""

    def __init__(self, base_loss: nn.Module, weights: Sequence[float]) -> None:
        super().__init__()
        if not weights or any(weight < 0 for weight in weights) or all(weight == 0 for weight in weights):
            raise ValueError(
                "deep-supervision weights must be non-negative and at least one must be positive"
            )
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weight_tensor / weight_tensor.sum(), persistent=True)
        self.base_loss = base_loss

    def forward(
        self,
        outputs: torch.Tensor | Sequence[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.base_loss(outputs, target)
        if len(outputs) != len(self.weights):
            raise ValueError(f"received {len(outputs)} outputs but {len(self.weights)} weights")

        total = target.new_zeros((), dtype=torch.float32)
        for output, weight in zip(outputs, self.weights):
            if weight.item() == 0:
                # Keep every returned head in the autograd graph. This produces
                # zero gradients for the disabled head and keeps DDP reduction
                # well-defined without find_unused_parameters overhead.
                total = total + output.sum() * 0.0
                continue
            scaled_target = target
            if output.shape[2:] != target.shape[2:]:
                scaled_target = F.interpolate(target.float(), size=output.shape[2:], mode="nearest")
            total = total + weight * self.base_loss(output, scaled_target)
        return total


def build_loss(loss_config: dict, number_of_outputs: int) -> nn.Module:
    base_loss = BraTSRegionLoss(
        dice_weight=loss_config.get("dice_weight", 1.0),
        bce_weight=loss_config.get("bce_weight", 1.0),
        hierarchy_weight=loss_config.get("hierarchy_weight", 0.0),
        smooth=loss_config.get("smooth", 1e-5),
        batch_dice=loss_config.get("batch_dice", False),
    )
    weights = loss_config.get("deep_supervision_weights")
    if weights is None:
        weights = [1.0 / (2**index) for index in range(number_of_outputs)]
        weights[-1] = 0.0
    if number_of_outputs == 1:
        return base_loss
    return DeepSupervisionLoss(base_loss, weights)
