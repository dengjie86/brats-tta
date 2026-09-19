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


class MultiClassDiceLoss(nn.Module):
    """Soft multiclass Dice for mutually-exclusive class-index targets."""

    def __init__(
        self,
        smooth: float = 1e-5,
        batch_dice: bool = False,
        include_background: bool = True,
    ) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.batch_dice = bool(batch_dice)
        self.include_background = bool(include_background)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5:
            raise ValueError(f"expected logits [B,C,D,H,W], got {tuple(logits.shape)}")
        if target.ndim != 4 or tuple(target.shape) != (logits.shape[0], *logits.shape[2:]):
            raise ValueError(
                f"multiclass target must have shape [B,D,H,W] matching logits, got {tuple(target.shape)}"
            )
        target = target.long()
        if target.numel() and (target.min() < 0 or target.max() >= logits.shape[1]):
            raise ValueError("class-index target contains an invalid class")
        probabilities = torch.softmax(logits.float(), dim=1)
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        start = 0 if self.include_background else 1
        probabilities = probabilities[:, start:]
        one_hot = one_hot[:, start:]
        reduce_dimensions = (0, 2, 3, 4) if self.batch_dice else (2, 3, 4)
        intersection = (probabilities * one_hot).sum(dim=reduce_dimensions)
        denominator = probabilities.sum(dim=reduce_dimensions) + one_hot.sum(dim=reduce_dimensions)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BraTSClassLoss(nn.Module):
    """Cross-entropy plus multiclass Dice for the four-class softmax head."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        batch_dice: bool = False,
        include_background: bool = True,
        class_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.dice = MultiClassDiceLoss(
            smooth=smooth, batch_dice=batch_dice, include_background=include_background
        )
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 4:
            raise ValueError("BraTSClassLoss expects class-index target [B,D,H,W]")
        weights = self.class_weights
        if weights is not None and weights.numel() != logits.shape[1]:
            raise ValueError("class_weights length must match logits channels")
        loss = self.dice_weight * self.dice(logits, target)
        loss = loss + self.ce_weight * F.cross_entropy(logits.float(), target.long(), weight=weights)
        return loss


class DeepSupervisionLoss(nn.Module):
    """Apply a region or class loss to high-to-low-resolution decoder outputs."""

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
                if target.ndim == 4:
                    scaled_target = F.interpolate(
                        target[:, None].float(), size=output.shape[2:], mode="nearest"
                    ).squeeze(1).long()
                else:
                    scaled_target = F.interpolate(target.float(), size=output.shape[2:], mode="nearest")
            total = total + weight * self.base_loss(output, scaled_target)
        return total


def build_loss(
    loss_config: dict,
    number_of_outputs: int,
    output_mode: str = "regions_sigmoid",
) -> nn.Module:
    if output_mode == "classes_softmax":
        base_loss: nn.Module = BraTSClassLoss(
            dice_weight=loss_config.get("dice_weight", 1.0),
            ce_weight=loss_config.get("ce_weight", loss_config.get("bce_weight", 1.0)),
            smooth=loss_config.get("smooth", 1e-5),
            batch_dice=loss_config.get("batch_dice", False),
            include_background=loss_config.get("include_background", True),
            class_weights=loss_config.get("class_weights"),
        )
    elif output_mode == "regions_sigmoid":
        base_loss = BraTSRegionLoss(
            dice_weight=loss_config.get("dice_weight", 1.0),
            bce_weight=loss_config.get("bce_weight", 1.0),
            hierarchy_weight=loss_config.get("hierarchy_weight", 0.0),
            smooth=loss_config.get("smooth", 1e-5),
            batch_dice=loss_config.get("batch_dice", False),
        )
    else:
        raise ValueError(f"unknown output_mode {output_mode!r}")
    weights = loss_config.get("deep_supervision_weights")
    if weights is None:
        weights = [1.0 / (2**index) for index in range(number_of_outputs)]
        weights[-1] = 0.0
    if number_of_outputs == 1:
        return base_loss
    return DeepSupervisionLoss(base_loss, weights)
