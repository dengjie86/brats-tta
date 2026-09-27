from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from brats_tta.tta.tent import TentStepResult

AdaptationScope = Literal["norm", "full"]


def _unwrap_logits(value: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    return value[0] if isinstance(value, (tuple, list)) else value


def _classes_to_regions(classes: torch.Tensor) -> torch.Tensor:
    if classes.ndim != 4:
        raise ValueError("class predictions must have shape [B,D,H,W]")
    et = classes == 3
    tc = (classes == 1) | et
    wt = classes > 0
    return torch.stack((et, tc, wt), dim=1)


def compute_adic_quality(
    reference_logits: torch.Tensor,
    dropout_probabilities: torch.Tensor,
) -> dict[str, float]:
    """Estimate ET/TC/WT quality using dropout agreement and confidence."""

    if reference_logits.ndim != 5 or reference_logits.shape[1] != 4:
        raise ValueError("ADIC requires four-class [B,C,D,H,W] reference logits")
    if dropout_probabilities.ndim != 6 or dropout_probabilities.shape[2] != 4:
        raise ValueError("dropout probabilities must have shape [S,B,4,D,H,W]")
    if tuple(dropout_probabilities.shape[1:]) != tuple(reference_logits.shape):
        raise ValueError("dropout and reference prediction shapes do not match")

    reference_classes = reference_logits.argmax(dim=1)
    reference_regions = _classes_to_regions(reference_classes)
    dropout_classes = dropout_probabilities.argmax(dim=2)
    dropout_regions = torch.stack(
        [_classes_to_regions(sample) for sample in dropout_classes],
        dim=0,
    )
    reduce_dimensions = (3, 4, 5)
    intersection = (dropout_regions & reference_regions.unsqueeze(0)).sum(
        dim=reduce_dimensions,
    ).float()
    denominator = (
        dropout_regions.sum(dim=reduce_dimensions).float()
        + reference_regions.unsqueeze(0).sum(dim=reduce_dimensions).float()
    )
    dice = torch.where(denominator > 0, 2.0 * intersection / denominator, 1.0)

    frequency = dropout_regions.float().mean(dim=0)
    binary_entropy = -(
        frequency * frequency.clamp_min(1e-6).log()
        + (1.0 - frequency) * (1.0 - frequency).clamp_min(1e-6).log()
    )
    flat_entropy = binary_entropy.flatten(start_dim=2)
    minimum = flat_entropy.amin(dim=2, keepdim=True)
    maximum = flat_entropy.amax(dim=2, keepdim=True)
    normalized_entropy = ((flat_entropy - minimum) / (maximum - minimum + 2e-6)).view_as(
        binary_entropy
    )
    union = dropout_regions.any(dim=0)
    union_count = union.flatten(start_dim=2).sum(dim=2).float()
    confidence_sum = ((1.0 - normalized_entropy) * union).flatten(start_dim=2).sum(dim=2)
    confidence_weight = torch.where(
        union_count > 0,
        (confidence_sum + 1e-6) / (union_count + 1e-6),
        1.0,
    )
    region_quality = confidence_weight * dice.mean(dim=0)

    mean_probabilities = dropout_probabilities.mean(dim=0)
    mismatch = mean_probabilities.argmax(dim=1) != reference_classes
    entropy = -(
        mean_probabilities * mean_probabilities.clamp_min(1e-8).log()
    ).sum(dim=1).mean()
    names = ("ET", "TC", "WT")
    values = region_quality.mean(dim=0)
    metrics = {f"adic_{name}": float(values[index].item()) for index, name in enumerate(names)}
    metrics["adic_mean"] = float(values.mean().item())
    metrics["adic_mismatch_fraction"] = float(mismatch.float().mean().item())
    metrics["adic_predictive_entropy"] = float(entropy.item())
    return metrics


class FeatureBank:
    def __init__(self, *, capacity: int = 10) -> None:
        if capacity < 1:
            raise ValueError("feature-bank capacity must be positive")
        self.capacity = int(capacity)
        self.features: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.features)

    def reset(self) -> None:
        self.features.clear()

    def push(self, feature: torch.Tensor) -> None:
        if len(self.features) >= self.capacity:
            self.features.pop(0)
        self.features.append(feature.detach().float().cpu().reshape(-1).clone())

    def similarity_weighted_reference(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if not self.features:
            return feature, 1.0
        query = feature.detach().float().reshape(-1)
        candidates = torch.stack(self.features).to(query.device)
        similarities = F.cosine_similarity(candidates, query.unsqueeze(0), dim=1)
        weights = similarities.softmax(dim=0)
        reference = (candidates * weights.unsqueeze(1)).sum(dim=0)
        weighted_similarity = float((similarities * weights).sum().item())
        return reference.view_as(feature), weighted_similarity

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "features": self.features}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("TEGDA feature-bank capacity does not match saved state")
        self.features = [feature.detach().float().cpu().clone() for feature in state["features"]]
        if len(self.features) > self.capacity:
            raise ValueError("invalid TEGDA feature-bank state")


def _configure_parameters(
    model: nn.Module,
    *,
    scope: AdaptationScope,
) -> tuple[list[nn.Parameter], list[str]]:
    model.eval()
    if scope == "full":
        model.requires_grad_(True)
        parameters = list(model.parameters())
        names = [name for name, _ in model.named_parameters()]
    elif scope == "norm":
        model.requires_grad_(False)
        parameters = []
        names = []
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.BatchNorm3d) or not module.affine:
                continue
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            for parameter_name in ("weight", "bias"):
                parameter = getattr(module, parameter_name)
                if parameter is not None:
                    parameter.requires_grad_(True)
                    parameters.append(parameter)
                    names.append(f"{module_name}.{parameter_name}")
    else:
        raise ValueError(f"unknown TEGDA adaptation scope: {scope}")
    if not parameters:
        raise ValueError("TEGDA found no parameters to adapt")
    return parameters, names


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _symmetric_soft_cross_entropy(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    class_count = student.shape[1]
    normalization = torch.log(
        torch.tensor(float(class_count), device=student.device, dtype=student.dtype)
    )
    target_probabilities = target.softmax(dim=1)
    student_probabilities = student.softmax(dim=1)
    forward = -(target_probabilities * student.log_softmax(dim=1)).sum(dim=1).mean()
    reverse = -(student_probabilities * target.log_softmax(dim=1)).sum(dim=1).mean()
    return 0.5 * (forward + reverse) / normalization


class TEGDAAdapter:
    """Memory-conscious 3D compatibility port of TEGDA for the plain U-Net."""

    method = "tegda"

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float = 1e-4,
        betas: tuple[float, float] = (0.5, 0.999),
        dropout_samples: int = 10,
        dropout_probability: float = 0.5,
        feature_bank_size: int = 10,
        feature_percentile: float = 0.90,
        scope: AdaptationScope = "full",
        steps: int = 1,
    ) -> None:
        if getattr(model, "output_mode", None) != "classes_softmax":
            raise ValueError("TEGDA requires the four-class softmax source model")
        if not hasattr(model, "encode_features") or not hasattr(model, "decode_features"):
            raise ValueError("TEGDA requires a model exposing encode_features/decode_features")
        if learning_rate <= 0 or dropout_samples < 2 or steps < 1:
            raise ValueError("invalid TEGDA optimization settings")
        if not 0 < dropout_probability < 1:
            raise ValueError("dropout_probability must be in (0, 1)")
        if not 0 <= feature_percentile <= 1:
            raise ValueError("feature_percentile must be in [0, 1]")
        self.model = model
        self.scope = scope
        self.parameters, self.parameter_names = _configure_parameters(model, scope=scope)
        self.learning_rate = float(learning_rate)
        self.betas = tuple(float(value) for value in betas)
        self.dropout_samples = int(dropout_samples)
        self.dropout_probability = float(dropout_probability)
        self.feature_percentile = float(feature_percentile)
        self.steps = int(steps)
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=0.0,
        )
        self.teacher = deepcopy(self.model).eval()
        self.teacher.requires_grad_(False)
        self.feature_bank = FeatureBank(capacity=feature_bank_size)
        self.quality_history: list[float] = []
        self._source_parameters = [parameter.detach().cpu().clone() for parameter in self.parameters]
        self._source_optimizer_state = deepcopy(self.optimizer.state_dict())
        self._source_teacher_state = _cpu_state_dict(self.teacher)
        self.total_updates = 0
        self.begin_case()

    def begin_case(self) -> None:
        self._case_updates = 0
        self._case_quality: list[float] = []
        self._case_losses: list[float] = []
        self._case_teacher_losses: list[float] = []
        self._case_refinement_losses: list[float] = []
        self._case_similarities: list[float] = []
        self._case_adic: list[dict[str, float]] = []

    @torch.no_grad()
    def reset(self) -> None:
        for parameter, source in zip(self.parameters, self._source_parameters):
            parameter.copy_(source.to(parameter))
        self.optimizer.load_state_dict(deepcopy(self._source_optimizer_state))
        self.teacher.load_state_dict(self._source_teacher_state, strict=True)
        self.feature_bank.reset()
        self.quality_history.clear()
        self.total_updates = 0

    @torch.no_grad()
    def _quality_and_refined_logits(
        self,
        images: torch.Tensor,
    ) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, float]:
        self.model.eval()
        features = self.model.encode_features(images)
        reference_logits = _unwrap_logits(self.model.decode_features(features))
        stochastic_probabilities: list[torch.Tensor] = []
        for _ in range(self.dropout_samples):
            stochastic_features = list(features)
            stochastic_features[-1] = F.dropout3d(
                features[-1],
                p=self.dropout_probability,
                training=True,
            )
            stochastic_logits = _unwrap_logits(self.model.decode_features(stochastic_features))
            stochastic_probabilities.append(stochastic_logits.softmax(dim=1))
        dropout_probabilities = torch.stack(stochastic_probabilities, dim=0)
        adic = compute_adic_quality(reference_logits, dropout_probabilities)
        quality = adic["adic_mean"]

        bottleneck = features[-1]
        refined = torch.empty_like(bottleneck)
        similarities: list[float] = []
        for batch_index in range(bottleneck.shape[0]):
            reference, similarity = self.feature_bank.similarity_weighted_reference(
                bottleneck[batch_index]
            )
            refined[batch_index] = (
                quality * bottleneck[batch_index]
                + (1.0 - quality) * reference.to(bottleneck)
            )
            similarities.append(similarity)
        refined_features = list(features)
        refined_features[-1] = refined
        refined_logits = _unwrap_logits(self.model.decode_features(refined_features)).detach()
        mean_similarity = float(sum(similarities) / len(similarities))
        return adic, refined_logits, bottleneck.detach(), mean_similarity

    @torch.no_grad()
    def _update_teacher(self, quality: float) -> None:
        for teacher_parameter, student_parameter in zip(
            self.teacher.parameters(),
            self.model.parameters(),
        ):
            teacher_parameter.mul_(1.0 - quality).add_(student_parameter, alpha=quality)

    @torch.no_grad()
    def _update_feature_bank(self, bottleneck: torch.Tensor, quality: float) -> None:
        threshold = None
        if self.quality_history:
            threshold = float(
                torch.quantile(
                    torch.tensor(self.quality_history, dtype=torch.float32),
                    self.feature_percentile,
                ).item()
            )
        if len(self.feature_bank) == 0 or threshold is None or quality >= threshold:
            for feature in bottleneck:
                self.feature_bank.push(feature)
        self.quality_history.append(quality)

    @torch.enable_grad()
    def predict_and_adapt(self, images: torch.Tensor) -> TentStepResult:
        loss_value = float("nan")
        for _ in range(self.steps):
            adic, refined_logits, bottleneck, similarity = self._quality_and_refined_logits(images)
            quality = adic["adic_mean"]
            with torch.no_grad():
                teacher_logits = _unwrap_logits(self.teacher(images)).detach()

            self.optimizer.zero_grad(set_to_none=True)
            student_logits = _unwrap_logits(self.model(images))
            teacher_loss = _symmetric_soft_cross_entropy(student_logits, teacher_logits)
            refinement_loss = _symmetric_soft_cross_entropy(student_logits, refined_logits)
            loss = teacher_loss + quality * refinement_loss
            if not torch.isfinite(loss):
                raise RuntimeError("TEGDA compatibility loss became non-finite")
            loss.backward()
            self.optimizer.step()
            self._update_teacher(quality)
            self._update_feature_bank(bottleneck, quality)

            loss_value = float(loss.detach().item())
            self._case_updates += 1
            self.total_updates += 1
            self._case_quality.append(quality)
            self._case_losses.append(loss_value)
            self._case_teacher_losses.append(float(teacher_loss.detach().item()))
            self._case_refinement_losses.append(float(refinement_loss.detach().item()))
            self._case_similarities.append(similarity)
            self._case_adic.append(adic)

        with torch.no_grad():
            logits = _unwrap_logits(self.model(images)).detach().float()
        return TentStepResult(logits=logits, entropy=loss_value)

    def case_diagnostics(self) -> dict[str, float | int | None]:
        diagnostics: dict[str, float | int | None] = {
            "tegda_updates": self._case_updates,
            "tegda_quality": _mean_or_none(self._case_quality),
            "tegda_loss": _mean_or_none(self._case_losses),
            "tegda_teacher_loss": _mean_or_none(self._case_teacher_losses),
            "tegda_refinement_loss": _mean_or_none(self._case_refinement_losses),
            "tegda_feature_similarity": _mean_or_none(self._case_similarities),
            "tegda_feature_bank_size": len(self.feature_bank),
            "tegda_total_updates": self.total_updates,
        }
        for key in (
            "adic_ET",
            "adic_TC",
            "adic_WT",
            "adic_mean",
            "adic_mismatch_fraction",
            "adic_predictive_entropy",
        ):
            diagnostics[f"tegda_{key}"] = _mean_or_none(
                [values[key] for values in self._case_adic]
            )
        return diagnostics

    def adaptation_state_dict(self) -> dict[str, Any]:
        return {
            "parameters": [parameter.detach().cpu() for parameter in self.parameters],
            "optimizer": self.optimizer.state_dict(),
            "teacher": _cpu_state_dict(self.teacher),
            "feature_bank": self.feature_bank.state_dict(),
            "quality_history": self.quality_history,
            "total_updates": self.total_updates,
        }

    @torch.no_grad()
    def load_adaptation_state_dict(self, state: dict[str, Any]) -> None:
        values = state["parameters"]
        if len(values) != len(self.parameters):
            raise ValueError("TEGDA saved parameter count does not match model")
        for parameter, value in zip(self.parameters, values):
            parameter.copy_(value.to(parameter))
        self.optimizer.load_state_dict(state["optimizer"])
        self.teacher.load_state_dict(state["teacher"], strict=True)
        self.feature_bank.load_state_dict(state["feature_bank"])
        self.quality_history = [float(value) for value in state["quality_history"]]
        self.total_updates = int(state["total_updates"])

    def method_metadata(self) -> dict[str, Any]:
        return {
            "implementation": "tegda_plain_unet3d_compat_v2",
            "reference_paper": "https://papers.miccai.org/miccai-2025/paper/2263_paper.pdf",
            "reference_repository": "HiLab-git/TEGDA@2f515079a69f8e7312084ab6974162afb5622315",
            "learning_rate": self.learning_rate,
            "optimizer": "Adam",
            "betas": list(self.betas),
            "weight_decay": 0.0,
            "adaptation_scope": self.scope,
            "adapted_parameter_count": sum(parameter.numel() for parameter in self.parameters),
            "adapted_parameter_names": self.parameter_names,
            "dropout_samples": self.dropout_samples,
            "dropout_probability": self.dropout_probability,
            "dropout_location": "bottleneck_feature",
            "feature_bank_size": self.feature_bank.capacity,
            "feature_percentile": self.feature_percentile,
            "feature_fusion": "cosine_softmax_all_bank_entries",
            "consistency_objective": "symmetric_soft_cross_entropy",
            "quality_guided_teacher_update": "teacher=(1-quality)*teacher+quality*student",
            "steps_per_patch_batch": self.steps,
            "default_protocol": "paper_hyperparameters_project_sliding_window_protocol",
            "exact_reference_reproduction": False,
            "compatibility_notes": [
                "the source plain U-Net has no trained dropout; stochastic dropout is "
                "applied at its bottleneck",
                "the released 3D TEGDA imports a 2D feature-update module; this port "
                "uses 3D bottleneck features",
                "ADIC follows the released 3D region-wise hard-prediction implementation "
                "rather than the paper's class-probability soft-Dice equation",
                "the released code uses symmetric soft cross-entropy although the paper "
                "describes Dice plus cross-entropy",
                "paper hyperparameters are retained, but native-grid sliding-window "
                "adaptation is used to match this project's Source and TENT evaluations",
                "refined and teacher predictions are detached targets to keep one adaptation graph in memory",
                "full scope follows the paper's all-parameter adaptation setting; norm scope "
                "is retained only as an explicit memory-saving ablation",
            ],
        }


def _mean_or_none(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None
