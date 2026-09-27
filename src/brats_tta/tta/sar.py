from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from brats_tta.tta.tent import TentStepResult, categorical_prediction_entropy, configure_tent


class SharpnessAwareOptimizer:
    """Two-step SAM optimizer used by SAR."""

    def __init__(
        self,
        parameters: list[nn.Parameter],
        *,
        learning_rate: float,
        momentum: float = 0.9,
        rho: float = 0.05,
    ) -> None:
        if not parameters:
            raise ValueError("SAR requires trainable normalization parameters")
        if learning_rate <= 0 or momentum < 0 or rho < 0:
            raise ValueError("invalid SAR optimizer settings")
        self.parameters = parameters
        self.rho = float(rho)
        self.base_optimizer = torch.optim.SGD(
            parameters,
            lr=float(learning_rate),
            momentum=float(momentum),
        )
        self._perturbations: dict[nn.Parameter, torch.Tensor] = {}

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def first_step(self) -> None:
        gradients = [parameter.grad for parameter in self.parameters if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("SAM first step requires gradients")
        device = gradients[0].device
        gradient_norm = torch.linalg.vector_norm(
            torch.stack([gradient.norm(p=2).to(device) for gradient in gradients]),
            ord=2,
        )
        scale = self.rho / (gradient_norm + 1e-12)
        self._perturbations.clear()
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            perturbation = parameter.grad * scale.to(parameter)
            parameter.add_(perturbation)
            self._perturbations[parameter] = perturbation
        self.zero_grad()

    @torch.no_grad()
    def second_step(self, *, update: bool) -> None:
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
            raise RuntimeError("cannot save SAR state between SAM steps")
        return {
            "rho": self.rho,
            "base_optimizer": self.base_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not math.isclose(float(state["rho"]), self.rho):
            raise ValueError("SAR rho does not match saved state")
        self.base_optimizer.load_state_dict(state["base_optimizer"])


class SARAdapter:
    """3D segmentation port of sharpness-aware reliable entropy minimization.

    The official SAR objective filters classification samples. For dense BraTS
    predictions, the released TEGDA comparison applies the same rule per voxel;
    this adapter follows that convention for the four-class softmax model.
    """

    method = "sar"

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        rho: float = 0.05,
        entropy_margin: float = 0.4 * math.log(4.0),
        recovery_threshold: float | None = 0.2,
        steps: int = 1,
    ) -> None:
        if steps < 1:
            raise ValueError("SAR steps must be positive")
        if entropy_margin <= 0:
            raise ValueError("SAR entropy margin must be positive")
        if recovery_threshold is not None and recovery_threshold < 0:
            raise ValueError("SAR recovery threshold must be non-negative")
        self.model = model
        self.parameters, self.parameter_names = configure_tent(model)
        if getattr(model, "output_mode", None) != "classes_softmax":
            raise ValueError("SAR requires the four-class softmax source model")
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.rho = float(rho)
        self.entropy_margin = float(entropy_margin)
        self.recovery_threshold = recovery_threshold
        self.steps = int(steps)
        self._source_parameters = [parameter.detach().clone() for parameter in self.parameters]
        self.optimizer = SharpnessAwareOptimizer(
            self.parameters,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
        )
        self._source_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.ema_entropy: float | None = None
        self.total_resets = 0
        self.begin_case()

    def begin_case(self) -> None:
        self._case_updates = 0
        self._case_skipped = 0
        self._case_resets = 0
        self._case_reliable_fractions: list[float] = []
        self._case_objectives: list[float] = []

    @torch.no_grad()
    def reset(self) -> None:
        for parameter, source in zip(self.parameters, self._source_parameters):
            parameter.copy_(source)
        self.optimizer.load_state_dict(deepcopy(self._source_optimizer_state))
        self.ema_entropy = None

    @torch.enable_grad()
    def predict_and_adapt(self, images: torch.Tensor) -> TentStepResult:
        output_for_stitching: torch.Tensor | None = None
        objective = float("nan")
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            logits = self.model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            output_for_stitching = logits.detach().float()
            entropy = categorical_prediction_entropy(logits)
            reliable = entropy < self.entropy_margin
            reliable_fraction = float(reliable.float().mean().detach().item())
            self._case_reliable_fractions.append(reliable_fraction)
            if not reliable.any():
                objective = float(entropy.mean().detach().item())
                self._case_objectives.append(objective)
                self._case_skipped += 1
                continue

            first_loss = entropy[reliable].mean()
            first_loss.backward()
            self.optimizer.first_step()

            second_logits = self.model(images)
            if isinstance(second_logits, (tuple, list)):
                second_logits = second_logits[0]
            second_candidates = categorical_prediction_entropy(second_logits)[reliable]
            second_reliable = second_candidates < self.entropy_margin
            if not second_reliable.any():
                self.optimizer.second_step(update=False)
                objective = float(first_loss.detach().item())
                self._case_objectives.append(objective)
                self._case_skipped += 1
                continue

            second_loss = second_candidates[second_reliable].mean()
            objective = float(second_loss.detach().item())
            second_loss.backward()
            self.optimizer.second_step(update=True)
            self._case_updates += 1
            self._case_objectives.append(objective)
            self.ema_entropy = (
                objective
                if self.ema_entropy is None
                else 0.9 * self.ema_entropy + 0.1 * objective
            )
            if (
                self.recovery_threshold is not None
                and self.ema_entropy < self.recovery_threshold
            ):
                self.reset()
                self._case_resets += 1
                self.total_resets += 1

        assert output_for_stitching is not None
        return TentStepResult(logits=output_for_stitching, entropy=objective)

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
            "sar_total_resets": self.total_resets,
        }

    def adaptation_state_dict(self) -> dict[str, Any]:
        return {
            "parameters": [parameter.detach().cpu() for parameter in self.parameters],
            "optimizer": self.optimizer.state_dict(),
            "ema_entropy": self.ema_entropy,
            "total_resets": self.total_resets,
        }

    @torch.no_grad()
    def load_adaptation_state_dict(self, state: dict[str, Any]) -> None:
        values = state["parameters"]
        if len(values) != len(self.parameters):
            raise ValueError("SAR saved parameter count does not match model")
        for parameter, value in zip(self.parameters, values):
            parameter.copy_(value.to(parameter))
        self.optimizer.load_state_dict(state["optimizer"])
        self.ema_entropy = state["ema_entropy"]
        self.total_resets = int(state["total_resets"])

    def method_metadata(self) -> dict[str, Any]:
        return {
            "implementation": "sar_dense_softmax_3d_v1",
            "reference_repository": "mr-eggplant/SAR@20f6e24b17525f34503510afccedc0629b67b7c4",
            "reference_behavior": "TEGDA BraTS SAR comparison",
            "optimizer": "SAM_SGD",
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "rho": self.rho,
            "entropy_margin": self.entropy_margin,
            "recovery_threshold": self.recovery_threshold,
            "updated_parameters": "batch_norm_affine",
            "reliability_unit": "voxel",
            "steps_per_patch_batch": self.steps,
            "exact_reference_reproduction": False,
            "compatibility_notes": [
                "classification sample filtering is applied per voxel",
                "source plain UNet and native sliding-window inference are retained",
            ],
        }
