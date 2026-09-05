from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def binary_prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Bernoulli entropy for the independent ET/TC/WT sigmoid outputs."""

    probabilities = torch.sigmoid(logits.float()).clamp(1e-7, 1.0 - 1e-7)
    return -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )


def configure_tent(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    """Freeze the network except InstanceNorm affine scale and shift parameters.

    InstanceNorm3d with ``track_running_stats=False`` uses current-instance
    statistics in both train and eval modes.  Keeping eval mode also disables
    deep-supervision outputs while preserving the statistics required by Tent.
    """

    model.eval()
    model.requires_grad_(False)
    parameters: list[nn.Parameter] = []
    names: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.InstanceNorm3d) or not module.affine:
            continue
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module, parameter_name)
            if parameter is None:
                continue
            parameter.requires_grad_(True)
            parameters.append(parameter)
            names.append(f"{module_name}.{parameter_name}")
    if not parameters:
        raise ValueError("Tent requires affine InstanceNorm3d parameters")
    return parameters, names


def configure_norm_stats(model: nn.Module) -> dict[str, int | bool]:
    """Configure the statistics-only baseline and describe its effective state.

    The source model has no persistent normalization buffers.  Consequently,
    statistics-only adaptation is already performed by every source forward and
    is mathematically identical to source-only inference.
    """

    model.eval()
    model.requires_grad_(False)
    layers = [module for module in model.modules() if isinstance(module, nn.InstanceNorm3d)]
    tracked_layers = [module for module in layers if module.track_running_stats]
    return {
        "instance_norm_layers": len(layers),
        "tracked_stat_layers": len(tracked_layers),
        "equivalent_to_source": len(tracked_layers) == 0,
    }


@dataclass(frozen=True)
class TentStepResult:
    logits: torch.Tensor
    entropy: float


class TentAdapter:
    """One-step online Tent updates with optional episodic parameter resets."""

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float = 1e-3,
        steps: int = 1,
        use_amp: bool = True,
        brain_mask: bool = False,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("Tent learning rate must be positive")
        if steps <= 0:
            raise ValueError("Tent steps must be positive")
        self.model = model
        self.parameters, self.parameter_names = configure_tent(model)
        self.learning_rate = float(learning_rate)
        self.steps = int(steps)
        self.use_amp = bool(use_amp)
        self.brain_mask = bool(brain_mask)
        self._source_parameters = [parameter.detach().clone() for parameter in self.parameters]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    @torch.no_grad()
    def reset(self) -> None:
        for parameter, source in zip(self.parameters, self._source_parameters):
            parameter.copy_(source)
        self.optimizer.state.clear()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.enable_grad()
    def predict_and_adapt(self, images: torch.Tensor) -> TentStepResult:
        output_for_stitching: torch.Tensor | None = None
        entropy_value = 0.0
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            amp_enabled = bool(self.use_amp and images.device.type == "cuda")
            with torch.autocast(
                device_type=images.device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = self.model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
            if output_for_stitching is None:
                output_for_stitching = logits.detach().float()
            entropy = binary_prediction_entropy(logits)
            if self.brain_mask:
                foreground = images.detach().abs().sum(dim=1, keepdim=True) > 0
                mask = foreground.expand_as(entropy)
                loss = entropy[mask].mean() if mask.any() else entropy.mean()
            else:
                loss = entropy.mean()
            loss.backward()
            self.optimizer.step()
            entropy_value = float(loss.detach().item())
        assert output_for_stitching is not None
        return TentStepResult(logits=output_for_stitching, entropy=entropy_value)
