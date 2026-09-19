"""Compatible SAR and CoTTA adapters for the project's sigmoid/IN source model.

These implementations preserve the central update rules and the hyperparameters
used by the released TEGDA comparison code.  They deliberately do not claim to
be exact reproductions: the source model has independent ET/TC/WT Bernoulli
outputs and untracked InstanceNorm rather than a four-class softmax with BN.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from brats_tta.tta.dense_objectives import (
    ConfidenceReduction,
    DenseReduction,
    binary_prediction_entropy,
    dense_prediction_confidence,
    dense_teacher_consistency_loss,
    image_brain_mask,
)
from brats_tta.tta.tent import configure_tent


@dataclass(frozen=True)
class BaselineStepResult:
    logits: torch.Tensor
    entropy: float


class SharpnessAwareOptimizer:
    """Small SAM wrapper using SGD, matching TEGDA's SAR optimizer settings."""

    def __init__(
        self,
        parameters: list[nn.Parameter],
        *,
        learning_rate: float,
        momentum: float = 0.9,
        rho: float = 0.05,
    ) -> None:
        if not parameters:
            raise ValueError("SAM requires parameters")
        if learning_rate <= 0 or rho < 0:
            raise ValueError("Invalid SAM learning rate or rho")
        self.parameters = parameters
        self.rho = float(rho)
        self.base_optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)
        self._perturbations: dict[nn.Parameter, torch.Tensor] = {}

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self) -> None:
        gradients = [p.grad for p in self.parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError("SAM first step requires gradients")
        device = gradients[0].device
        norm = torch.linalg.vector_norm(
            torch.stack([gradient.norm(p=2).to(device) for gradient in gradients]), ord=2
        )
        scale = self.rho / (norm + 1e-12)
        self._perturbations.clear()
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            perturbation = parameter.grad * scale.to(parameter)
            parameter.add_(perturbation)
            self._perturbations[parameter] = perturbation
        self.zero_grad()

    @torch.no_grad()
    def second_step(self, *, update: bool = True) -> None:
        if not self._perturbations:
            raise RuntimeError("SAM second step called before first step")
        for parameter, perturbation in self._perturbations.items():
            parameter.sub_(perturbation)
        self._perturbations.clear()
        if update:
            self.base_optimizer.step()
        self.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        if self._perturbations:
            raise RuntimeError("Cannot checkpoint SAM between its two steps")
        return {"rho": self.rho, "base_optimizer": self.base_optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state["rho"]) != self.rho:
            raise ValueError("SAM rho mismatch")
        self.base_optimizer.load_state_dict(state["base_optimizer"])


class SARAdapter:
    """Sharpness-aware reliable entropy minimization for Bernoulli regions."""

    method = "sar"

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        rho: float = 0.05,
        entropy_margin: float = 0.4 * math.log(2.0),
        recovery_threshold: float | None = 0.02,
        objective_reduction: DenseReduction = "all",
        steps: int = 1,
    ) -> None:
        if steps < 1:
            raise ValueError("SAR steps must be positive")
        if entropy_margin <= 0:
            raise ValueError("SAR entropy margin must be positive")
        if recovery_threshold is not None and recovery_threshold < 0:
            raise ValueError("SAR recovery threshold must be nonnegative")
        self.model = model
        self.parameters, self.parameter_names = configure_tent(model)
        self._source_parameters = [p.detach().clone() for p in self.parameters]
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.rho = float(rho)
        self.entropy_margin = float(entropy_margin)
        self.recovery_threshold = recovery_threshold
        if objective_reduction not in {
            "all",
            "brain",
            "foreground_background_balanced",
        }:
            raise ValueError(f"unknown SAR objective reduction: {objective_reduction}")
        self.objective_reduction: DenseReduction = objective_reduction
        self.steps = int(steps)
        self.optimizer = SharpnessAwareOptimizer(
            self.parameters,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
        )
        self._source_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.ema_entropy: float | None = None
        self.total_resets = 0
        self.begin_case()

    def begin_case(self) -> None:
        self._case_resets = 0
        self._case_updates = 0
        self._case_skipped = 0
        self._case_reliable_fractions: list[float] = []
        self._case_objectives: list[float] = []

    @torch.no_grad()
    def reset(self) -> None:
        for parameter, source in zip(self.parameters, self._source_parameters):
            parameter.copy_(source)
        self.optimizer.load_state_dict(deepcopy(self._source_optimizer_state))
        self.ema_entropy = None

    @staticmethod
    def _voxel_entropy(logits: torch.Tensor) -> torch.Tensor:
        # Average the three independent Bernoulli entropies so the maximum is ln(2).
        return binary_prediction_entropy(logits).mean(dim=1)

    def _reliable_loss(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        *,
        initial_reliable: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, float]:
        """Map SAR's sample filtering to brain-supported dense predictions."""

        element_entropy = binary_prediction_entropy(logits)
        voxel_entropy = element_entropy.mean(dim=1)
        reliable = voxel_entropy < self.entropy_margin
        if self.objective_reduction == "all":
            support = torch.ones_like(reliable)
        else:
            support = image_brain_mask(images).squeeze(1)
            reliable = reliable & support
        if initial_reliable is not None:
            reliable = reliable & initial_reliable
        denominator = support.sum().clamp_min(1)
        reliable_fraction = float((reliable.sum() / denominator).detach().float().item())
        if not reliable.any():
            return None, reliable, reliable_fraction
        if self.objective_reduction != "foreground_background_balanced":
            return voxel_entropy[reliable].mean(), reliable, reliable_fraction

        # Avoid letting reliable background consume the entire SAR objective.
        probabilities = torch.sigmoid(logits.float()).detach()
        valid = reliable.unsqueeze(1).expand_as(element_entropy)
        group_means: list[torch.Tensor] = []
        for batch_index in range(element_entropy.shape[0]):
            for channel_index in range(element_entropy.shape[1]):
                for predicted_foreground in (True, False):
                    group = valid[batch_index, channel_index] & (
                        probabilities[batch_index, channel_index] >= 0.5
                        if predicted_foreground
                        else probabilities[batch_index, channel_index] < 0.5
                    )
                    if group.any():
                        group_means.append(element_entropy[batch_index, channel_index][group].mean())
        if not group_means:
            return None, reliable, reliable_fraction
        return torch.stack(group_means).mean(), reliable, reliable_fraction

    @torch.enable_grad()
    def predict_and_adapt(self, images: torch.Tensor) -> BaselineStepResult:
        output_for_stitching: torch.Tensor | None = None
        objective = float("nan")
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            logits = self.model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            output_for_stitching = logits.detach().float()
            first_loss, reliable, reliable_fraction = self._reliable_loss(logits, images)
            self._case_reliable_fractions.append(reliable_fraction)
            if first_loss is None:
                self._case_skipped += 1
                objective = float(self._voxel_entropy(logits).mean().detach().item())
                self._case_objectives.append(objective)
                continue

            first_loss.backward()
            self.optimizer.first_step()

            second_logits = self.model(images)
            if isinstance(second_logits, (tuple, list)):
                second_logits = second_logits[0]
            second_loss, _, _ = self._reliable_loss(second_logits, images, initial_reliable=reliable)
            if second_loss is None:
                self.optimizer.second_step(update=False)
                self._case_skipped += 1
                objective = float(first_loss.detach().item())
                self._case_objectives.append(objective)
                continue

            objective = float(second_loss.detach().item())
            second_loss.backward()
            self.optimizer.second_step(update=True)
            self._case_updates += 1
            self._case_objectives.append(objective)
            self.ema_entropy = (
                objective if self.ema_entropy is None else 0.9 * self.ema_entropy + 0.1 * objective
            )
            if self.recovery_threshold is not None and self.ema_entropy < self.recovery_threshold:
                self.reset()
                self._case_resets += 1
                self.total_resets += 1

        assert output_for_stitching is not None
        return BaselineStepResult(logits=output_for_stitching, entropy=objective)

    @property
    def post_model(self) -> nn.Module:
        return self.model

    def case_diagnostics(self) -> dict[str, float | int | None]:
        return {
            "sar_updates": self._case_updates,
            "sar_skipped_updates": self._case_skipped,
            "sar_recovery_resets": self._case_resets,
            "sar_reliable_fraction": (
                float(sum(self._case_reliable_fractions) / len(self._case_reliable_fractions))
                if self._case_reliable_fractions
                else 0.0
            ),
            "sar_objective": (
                float(sum(self._case_objectives) / len(self._case_objectives))
                if self._case_objectives
                else None
            ),
            "sar_ema_entropy": self.ema_entropy,
        }

    def continual_state_dict(self) -> dict[str, Any]:
        return {"ema_entropy": self.ema_entropy, "total_resets": self.total_resets}

    def load_continual_state_dict(self, state: dict[str, Any]) -> None:
        self.ema_entropy = state["ema_entropy"]
        self.total_resets = int(state["total_resets"])


def _random_legacy_augmentation(
    image: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, tuple[int, ...]]]:
    """Retain the first TEGDA-compatible port for controlled ablations."""

    rotation = int(torch.randint(0, 4, ()).item())
    augmented = torch.rot90(image, rotation, dims=(-2, -1))
    flip_dimensions = tuple(dimension for dimension in (-3, -2, -1) if bool(torch.rand(()) < 0.5))
    if flip_dimensions:
        augmented = torch.flip(augmented, flip_dimensions)
    scale = 0.9 + 0.2 * torch.rand((), device=image.device)
    shift = -0.1 + 0.2 * torch.rand((), device=image.device)
    augmented = augmented * scale + shift
    augmented = augmented + 0.005 * torch.randn_like(augmented)
    return augmented, (rotation, flip_dimensions)


def _random_source_compatible_augmentation(
    image: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, tuple[int, ...]]]:
    """Mirror the source-training augmentation with an invertible spatial part."""

    rotation = 0
    augmented = image.clone()
    flip_dimensions = tuple(dimension for dimension in (-3, -2, -1) if bool(torch.rand(()) < 0.5))
    if flip_dimensions:
        augmented = torch.flip(augmented, flip_dimensions)
    for channel_index in range(augmented.shape[1]):
        channel = augmented[:, channel_index]
        foreground = channel != 0
        if not foreground.any():
            continue
        scale = 0.9 + 0.2 * torch.rand((), device=image.device)
        shift = -0.1 + 0.2 * torch.rand((), device=image.device)
        channel[foreground] = channel[foreground] * scale + shift
        if torch.rand((), device=image.device) < 0.15:
            noise_std = 0.1 * torch.rand((), device=image.device)
            channel[foreground] += torch.randn_like(channel[foreground]) * noise_std
    return augmented, (rotation, flip_dimensions)


def _invert_spatial_augmentation(
    tensor: torch.Tensor, transform: tuple[int, tuple[int, ...]]
) -> torch.Tensor:
    rotation, flip_dimensions = transform
    if flip_dimensions:
        tensor = torch.flip(tensor, flip_dimensions)
    return torch.rot90(tensor, -rotation, dims=(-2, -1))


class CoTTAAdapter:
    """Mean-teacher CoTTA port with Bernoulli consistency and stochastic restore."""

    method = "cotta"

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.9,
        betas: tuple[float, float] = (0.9, 0.999),
        teacher_momentum: float = 0.99,
        restore_probability: float = 0.1,
        augmentation_threshold: float = 0.9,
        augmentation_count: int = 32,
        confidence_reduction: ConfidenceReduction = "all",
        confidence_uncertain_fraction: float = 0.01,
        consistency_reduction: DenseReduction = "all",
        minimum_consistency_mae: float = 0.0,
        augmentation_mode: str = "legacy_port",
        steps: int = 1,
    ) -> None:
        if steps < 1 or augmentation_count < 1:
            raise ValueError("CoTTA steps and augmentation count must be positive")
        if not 0 <= restore_probability <= 1:
            raise ValueError("CoTTA restore probability must be in [0, 1]")
        if not 0 <= teacher_momentum < 1:
            raise ValueError("CoTTA teacher momentum must be in [0, 1)")
        if not 0 < confidence_uncertain_fraction <= 1:
            raise ValueError("confidence_uncertain_fraction must be in (0, 1]")
        if minimum_consistency_mae < 0:
            raise ValueError("minimum_consistency_mae must be nonnegative")
        if augmentation_mode not in {"legacy_port", "source_training"}:
            raise ValueError(f"unknown CoTTA augmentation mode: {augmentation_mode}")
        self.model = model.eval()
        self.model.requires_grad_(True)
        named_parameters = list(self.model.named_parameters())
        self.parameter_names = [name for name, _ in named_parameters]
        self.parameters = [parameter for _, parameter in named_parameters]
        self._source_parameters = [p.detach().clone() for p in self.parameters]
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
        self._source_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.ema_model = deepcopy(self.model).eval().requires_grad_(False)
        self.anchor_model = deepcopy(self.model).eval().requires_grad_(False)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.betas = tuple(float(value) for value in betas)
        self.teacher_momentum = float(teacher_momentum)
        self.restore_probability = float(restore_probability)
        self.augmentation_threshold = float(augmentation_threshold)
        self.augmentation_count = int(augmentation_count)
        self.confidence_reduction = confidence_reduction
        self.confidence_uncertain_fraction = float(confidence_uncertain_fraction)
        self.consistency_reduction = consistency_reduction
        self.minimum_consistency_mae = float(minimum_consistency_mae)
        self.augmentation_mode = augmentation_mode
        self.steps = int(steps)
        self.begin_case()

    def begin_case(self) -> None:
        self._case_updates = 0
        self._case_skipped_updates = 0
        self._case_augmentations = 0
        self._case_confidences: list[float] = []
        self._case_losses: list[float] = []
        self._case_consistency_mae: list[float] = []
        self._case_restored = 0
        self._case_restore_total = 0

    @torch.no_grad()
    def reset(self) -> None:
        for parameter, source in zip(self.parameters, self._source_parameters):
            parameter.copy_(source)
        self.optimizer.load_state_dict(deepcopy(self._source_optimizer_state))
        self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
        self.anchor_model.load_state_dict(self.model.state_dict(), strict=True)

    @torch.no_grad()
    def _teacher_target(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, int]:
        anchor_logits = self.anchor_model(images)
        ema_logits = self.ema_model(images)
        assert isinstance(anchor_logits, torch.Tensor) and isinstance(ema_logits, torch.Tensor)
        confidence = float(
            dense_prediction_confidence(
                anchor_logits,
                images,
                reduction=self.confidence_reduction,
                uncertain_fraction=self.confidence_uncertain_fraction,
            ).item()
        )
        if confidence >= self.augmentation_threshold:
            return torch.sigmoid(ema_logits.float()), ema_logits.detach().float(), confidence, 0

        probabilities: list[torch.Tensor] = []
        augmentation = (
            _random_source_compatible_augmentation
            if self.augmentation_mode == "source_training"
            else _random_legacy_augmentation
        )
        for _ in range(self.augmentation_count):
            augmented, transform = augmentation(images)
            augmented_logits = self.ema_model(augmented)
            assert isinstance(augmented_logits, torch.Tensor)
            probability = torch.sigmoid(augmented_logits.float())
            probabilities.append(_invert_spatial_augmentation(probability, transform))
        target = torch.stack(probabilities).mean(dim=0).clamp(1e-7, 1.0 - 1e-7)
        target_logits = torch.logit(target)
        return target, target_logits, confidence, self.augmentation_count

    @torch.no_grad()
    def _update_teacher_and_restore(self) -> None:
        for ema_parameter, parameter in zip(self.ema_model.parameters(), self.parameters):
            ema_parameter.mul_(self.teacher_momentum).add_(parameter, alpha=1.0 - self.teacher_momentum)
        for parameter, source in zip(self.parameters, self._source_parameters):
            mask = torch.rand_like(parameter) < self.restore_probability
            self._case_restored += int(mask.sum().item())
            self._case_restore_total += mask.numel()
            parameter.copy_(torch.where(mask, source, parameter))

    @torch.enable_grad()
    def predict_and_adapt(self, images: torch.Tensor) -> BaselineStepResult:
        output_for_stitching: torch.Tensor | None = None
        loss_value = float("nan")
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            student_logits = self.model(images)
            if isinstance(student_logits, (tuple, list)):
                student_logits = student_logits[0]
            target, teacher_logits, confidence, augmentation_count = self._teacher_target(images)
            output_for_stitching = teacher_logits
            loss = dense_teacher_consistency_loss(
                student_logits,
                target,
                images,
                reduction=self.consistency_reduction,
                normalize=True,
            )
            probabilities = torch.sigmoid(student_logits.detach().float())
            brain = image_brain_mask(images).expand_as(probabilities)
            consistency_mae = float(
                (
                    (probabilities - target).abs()[brain].mean()
                    if brain.any()
                    else (probabilities - target).abs().mean()
                ).item()
            )
            self._case_consistency_mae.append(consistency_mae)
            self._case_augmentations += augmentation_count
            self._case_confidences.append(confidence)
            self._case_losses.append(float(loss.detach().item()))
            if self.minimum_consistency_mae > 0 and consistency_mae <= self.minimum_consistency_mae:
                self._case_skipped_updates += 1
                loss_value = float(loss.detach().item())
                continue
            loss.backward()
            self.optimizer.step()
            self._update_teacher_and_restore()
            loss_value = float(loss.detach().item())
            self._case_updates += 1
        assert output_for_stitching is not None
        return BaselineStepResult(logits=output_for_stitching, entropy=loss_value)

    @property
    def post_model(self) -> nn.Module:
        return self.ema_model

    def case_diagnostics(self) -> dict[str, float | int | None]:
        return {
            "cotta_updates": self._case_updates,
            "cotta_skipped_updates": self._case_skipped_updates,
            "cotta_augmentation_forwards": self._case_augmentations,
            "cotta_anchor_confidence": (
                float(sum(self._case_confidences) / len(self._case_confidences))
                if self._case_confidences
                else None
            ),
            "cotta_consistency_loss": (
                float(sum(self._case_losses) / len(self._case_losses)) if self._case_losses else None
            ),
            "cotta_consistency_probability_mae": (
                float(sum(self._case_consistency_mae) / len(self._case_consistency_mae))
                if self._case_consistency_mae
                else None
            ),
            "cotta_restore_fraction": (
                self._case_restored / self._case_restore_total if self._case_restore_total else 0.0
            ),
        }

    def continual_state_dict(self) -> dict[str, Any]:
        return {"ema_model": self.ema_model.state_dict()}

    def load_continual_state_dict(self, state: dict[str, Any]) -> None:
        self.ema_model.load_state_dict(state["ema_model"], strict=True)


def build_tegda_baseline_adapter(
    method: str,
    model: nn.Module,
    *,
    cotta_augmentations: int = 32,
    profile: str = "compatible",
) -> SARAdapter | CoTTAAdapter:
    if profile == "compatible":
        if method == "sar":
            return SARAdapter(model)
        if method == "cotta":
            return CoTTAAdapter(model, augmentation_count=cotta_augmentations)
    elif profile == "dense_v2":
        if method == "sar":
            return SARAdapter(
                model,
                recovery_threshold=None,
                objective_reduction="foreground_background_balanced",
            )
        if method == "cotta":
            return CoTTAAdapter(
                model,
                weight_decay=0.0,
                restore_probability=0.01,
                augmentation_count=cotta_augmentations,
                confidence_reduction="brain_uncertain",
                confidence_uncertain_fraction=0.01,
                consistency_reduction="foreground_background_balanced",
                minimum_consistency_mae=1e-7,
                augmentation_mode="source_training",
            )
    else:
        raise ValueError(f"Unsupported TTA baseline profile: {profile}")
    raise ValueError(f"Unsupported compatible TEGDA baseline: {method}")
