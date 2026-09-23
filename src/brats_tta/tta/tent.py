from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

TentBNScope = Literal["all", "encoder", "decoder", "shallow", "deep"]


def categorical_prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return voxel-wise categorical entropy for mutually exclusive classes."""

    if logits.ndim != 5 or logits.shape[1] < 2:
        raise ValueError("Tent requires [B,C,D,H,W] logits with C >= 2")
    log_probabilities = torch.log_softmax(logits.float(), dim=1)
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=1)


def _module_stage(module_name: str, branch: str) -> int | None:
    prefix = f"{branch}."
    if not module_name.startswith(prefix):
        return None
    stage_text = module_name[len(prefix) :].split(".", 1)[0]
    return int(stage_text) if stage_text.isdigit() else None


def _scope_matches(module_name: str, scope: TentBNScope) -> bool:
    """Return whether a BN layer belongs to the requested adaptation scope."""

    if scope == "all":
        return True
    if scope == "encoder":
        return module_name.startswith("encoder.")
    if scope == "decoder":
        return module_name.startswith("decoder.")

    encoder_stage = _module_stage(module_name, "encoder")
    decoder_stage = _module_stage(module_name, "decoder")
    if scope == "shallow":
        return (encoder_stage is not None and encoder_stage <= 2) or (
            decoder_stage is not None and decoder_stage >= 3
        )
    if scope == "deep":
        return (encoder_stage is not None and encoder_stage >= 3) or (
            decoder_stage is not None and decoder_stage <= 2
        )
    raise ValueError(f"unsupported Tent BN scope: {scope}")


def configure_tent(
    model: nn.Module,
    *,
    bn_scope: TentBNScope = "all",
    update_affine: bool = True,
    use_batch_stats: bool = True,
    allow_empty: bool = False,
) -> tuple[list[nn.Parameter], list[str]]:
    """Freeze the network except affine BatchNorm scale and shift parameters.

    BatchNorm running buffers can be disabled so target patch statistics are
    used, matching the original Tent configuration.  ``bn_scope`` selects the
    affine parameters that receive entropy gradients; statistics are configured
    independently for every BatchNorm layer.
    """

    model.eval()
    model.requires_grad_(False)
    parameters: list[nn.Parameter] = []
    names: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm3d) or not module.affine:
            continue
        if use_batch_stats:
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        if not update_affine or not _scope_matches(module_name, bn_scope):
            continue
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module, parameter_name)
            if parameter is None:
                continue
            parameter.requires_grad_(True)
            parameters.append(parameter)
            names.append(f"{module_name}.{parameter_name}")
    if update_affine and not parameters and not allow_empty:
        raise ValueError("Tent requires affine BatchNorm3d parameters in the selected scope")
    return parameters, names


@dataclass(frozen=True)
class TentStepResult:
    logits: torch.Tensor
    entropy: float


class TentAdapter:
    """Standard categorical TENT for the current BatchNorm source model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float = 1e-3,
        steps: int = 1,
        use_amp: bool = True,
        bn_scope: TentBNScope = "all",
        update_affine: bool = True,
        use_batch_stats: bool = True,
        allow_noop: bool = False,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("Tent learning rate must be positive")
        if steps <= 0:
            raise ValueError("Tent steps must be positive")
        self.model = model
        self.bn_scope = bn_scope
        self.update_affine = bool(update_affine)
        self.use_batch_stats = bool(use_batch_stats)
        self.parameters, self.parameter_names = configure_tent(
            model,
            bn_scope=bn_scope,
            update_affine=self.update_affine,
            use_batch_stats=self.use_batch_stats,
            allow_empty=allow_noop,
        )
        self.is_noop = not self.parameters
        self.output_mode = getattr(model, "output_mode", None)
        if self.output_mode != "classes_softmax":
            raise ValueError("Tent requires the current four-class softmax source model")
        self.learning_rate = float(learning_rate)
        self.steps = int(steps)
        self.use_amp = bool(use_amp)
        self._source_parameters = [parameter.detach().clone() for parameter in self.parameters]
        self.optimizer = (
            torch.optim.Adam(self.parameters, lr=self.learning_rate, weight_decay=0.0)
            if self.parameters
            else None
        )
        device_type = next(model.parameters()).device.type
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and device_type == "cuda" and bool(self.parameters)
        )
        self._source_scaler_state = self.scaler.state_dict().copy()

    @torch.no_grad()
    def reset(self) -> None:
        for parameter, source in zip(self.parameters, self._source_parameters):
            parameter.copy_(source)
        if self.optimizer is not None:
            self.optimizer.state.clear()
            self.optimizer.zero_grad(set_to_none=True)
        self.scaler.load_state_dict(self._source_scaler_state)

    @torch.enable_grad()
    def predict_and_adapt(self, images: torch.Tensor) -> TentStepResult:
        if not self.parameters:
            with torch.no_grad():
                logits = self.model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                entropy = categorical_prediction_entropy(logits).mean()
            return TentStepResult(logits=logits.detach().float(), entropy=float(entropy.item()))

        output_for_stitching: torch.Tensor | None = None
        entropy_value = 0.0
        for _ in range(self.steps):
            assert self.optimizer is not None
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
            # As in Tent's multi-step loop, return the last forward's output.
            # With one step this is the prediction before that step's update.
            output_for_stitching = logits.detach().float()
            loss = categorical_prediction_entropy(logits).mean()
            # Averaging over a full 128-cubed patch produces tiny per-logit gradients.
            # Computing entropy in float32 alone does not prevent underflow when
            # those gradients flow back into the float16 network outputs.
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            entropy_value = float(loss.detach().item())
        assert output_for_stitching is not None
        return TentStepResult(logits=output_for_stitching, entropy=entropy_value)
