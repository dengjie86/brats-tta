from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Sequence

import torch
from torch import nn

from brats_tta.tta.tent import TentStepResult


class AdaptiveBatchNorm3d(nn.BatchNorm3d):
    """3D version of VPTTA's source/target statistic blending layer."""

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        warm_n: float = 5.0,
    ) -> None:
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=True,
        )
        if warm_n <= 0:
            raise ValueError("warm_n must be positive")
        self.warm_n = float(warm_n)
        self.new_sample = False
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.long))
        self.bn_loss = torch.zeros(())

    @classmethod
    def from_batch_norm(cls, module: nn.BatchNorm3d, *, warm_n: float) -> AdaptiveBatchNorm3d:
        if module.running_mean is None or module.running_var is None:
            raise ValueError("VPTTA requires source BatchNorm3d running statistics")
        replacement = cls(
            module.num_features,
            eps=module.eps,
            momentum=float(module.momentum or 0.1),
            affine=module.affine,
            warm_n=warm_n,
        )
        reference = module.weight if module.weight is not None else module.running_mean
        replacement.to(device=reference.device, dtype=reference.dtype)
        replacement.load_state_dict(module.state_dict(), strict=False)
        replacement.train(module.training)
        return replacement

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(f"AdaptiveBatchNorm3d expects [B,C,D,H,W], got {inputs.shape}")
        if self.running_mean is None or self.running_var is None:
            raise RuntimeError("AdaptiveBatchNorm3d source statistics are unavailable")
        if self.new_sample:
            self.sample_count.add_(1)

        reduce_dimensions = (0, 2, 3, 4)
        current_mean = inputs.mean(dim=reduce_dimensions, keepdim=True).detach()
        current_var = inputs.var(dim=reduce_dimensions, unbiased=False, keepdim=True).detach()
        source_mean = self.running_mean.view(1, -1, 1, 1, 1)
        source_var = self.running_var.view(1, -1, 1, 1, 1)
        count = max(1, int(self.sample_count.item()))
        blend = 1.0 / (math.sqrt(count) / self.warm_n + 1.0)
        blended_mean = blend * current_mean + (1.0 - blend) * source_mean
        blended_var = blend * current_var + (1.0 - blend) * source_var

        sample_mean = inputs.mean(dim=(2, 3, 4), keepdim=True)
        sample_std = inputs.var(dim=(2, 3, 4), unbiased=False, keepdim=True).add(self.eps).sqrt()
        self.bn_loss = (
            (blended_mean - sample_mean).abs().mean()
            + (blended_var.add(self.eps).sqrt() - sample_std).abs().mean()
        )

        outputs = (inputs - blended_mean) / blended_var.add(self.eps).sqrt()
        if self.affine:
            assert self.weight is not None and self.bias is not None
            outputs = (
                outputs * self.weight.view(1, -1, 1, 1, 1)
                + self.bias.view(1, -1, 1, 1, 1)
            )
        return outputs


def convert_to_adaptive_batch_norm_3d(model: nn.Module, *, warm_n: float = 5.0) -> list[str]:
    """Replace every BatchNorm3d layer while preserving its source state."""

    converted: list[str] = []

    def convert(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, AdaptiveBatchNorm3d):
                converted.append(qualified_name)
            elif isinstance(child, nn.BatchNorm3d):
                setattr(
                    parent,
                    child_name,
                    AdaptiveBatchNorm3d.from_batch_norm(child, warm_n=warm_n),
                )
                converted.append(qualified_name)
            else:
                convert(child, qualified_name)

    convert(model, "")
    if not converted:
        raise ValueError("VPTTA requires BatchNorm3d layers with source running statistics")
    return converted


class FourierPrompt3D(nn.Module):
    """Learnable multiplicative prompt for centered 3D Fourier amplitudes."""

    def __init__(
        self,
        channels: int,
        patch_size: Sequence[int],
        *,
        prompt_alpha: float = 0.03,
    ) -> None:
        super().__init__()
        patch_size = tuple(int(value) for value in patch_size)
        if channels < 1:
            raise ValueError("channels must be positive")
        if len(patch_size) != 3 or any(value < 1 for value in patch_size):
            raise ValueError("patch_size must contain three positive values")
        if not 0 < prompt_alpha <= 1:
            raise ValueError("prompt_alpha must be in (0, 1]")
        self.channels = int(channels)
        self.patch_size = patch_size
        self.prompt_alpha = float(prompt_alpha)
        self.prompt_size = tuple(max(1, int(value * prompt_alpha)) for value in patch_size)
        self.data_prompt = nn.Parameter(torch.ones((1, channels, *self.prompt_size)))

    def _center_slices(self, spatial_shape: Sequence[int]) -> tuple[slice, slice, slice]:
        slices: list[slice] = []
        for full_size, prompt_size in zip(spatial_shape, self.prompt_size):
            if full_size != self.patch_size[len(slices)]:
                raise ValueError(
                    f"VPTTA prompt was built for {self.patch_size}, got {tuple(spatial_shape)}"
                )
            start = (full_size - prompt_size) // 2
            slices.append(slice(start, start + prompt_size))
        return tuple(slices)  # type: ignore[return-value]

    def low_frequency_amplitude(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},D,H,W], got {tuple(inputs.shape)}")
        spectrum = torch.fft.fftn(inputs.float(), dim=(-3, -2, -1))
        amplitude = torch.fft.fftshift(spectrum.abs(), dim=(-3, -2, -1))
        spatial_slices = self._center_slices(inputs.shape[2:])
        return amplitude[(slice(None), slice(None), *spatial_slices)]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 5 or inputs.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},D,H,W], got {tuple(inputs.shape)}")
        spectrum = torch.fft.fftn(inputs.float(), dim=(-3, -2, -1))
        phase = torch.angle(spectrum)
        shifted_amplitude = torch.fft.fftshift(spectrum.abs(), dim=(-3, -2, -1))
        spatial_slices = self._center_slices(inputs.shape[2:])
        low_frequency = shifted_amplitude[(slice(None), slice(None), *spatial_slices)]
        multiplier = torch.ones_like(shifted_amplitude)
        multiplier[(slice(None), slice(None), *spatial_slices)] = self.data_prompt
        prompted_amplitude = torch.fft.ifftshift(
            shifted_amplitude * multiplier,
            dim=(-3, -2, -1),
        )
        prompted_spectrum = torch.polar(prompted_amplitude, phase)
        prompted = torch.fft.ifftn(prompted_spectrum, dim=(-3, -2, -1)).real
        return prompted, low_frequency

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.shape != self.data_prompt.shape:
            raise ValueError(
                f"prompt initializer has shape {values.shape}, expected {self.data_prompt.shape}"
            )
        self.data_prompt.copy_(values.to(self.data_prompt))

    @torch.no_grad()
    def reset_ones(self) -> None:
        self.data_prompt.fill_(1.0)


class PromptMemoryBank:
    def __init__(self, *, capacity: int = 40, temperature: float = 0.2) -> None:
        if capacity < 1 or temperature <= 0:
            raise ValueError("invalid VPTTA memory settings")
        self.capacity = int(capacity)
        self.temperature = float(temperature)
        self.keys: list[torch.Tensor] = []
        self.prompts: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.keys)

    def reset(self) -> None:
        self.keys.clear()
        self.prompts.clear()

    def push(self, key: torch.Tensor, prompt: torch.Tensor) -> None:
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.prompts.pop(0)
        self.keys.append(key.detach().float().cpu().reshape(-1).clone())
        self.prompts.append(prompt.detach().float().cpu().clone())

    def nearest_prompt(self, key: torch.Tensor, *, neighbors: int) -> tuple[torch.Tensor, float]:
        if not self.keys:
            raise RuntimeError("cannot query an empty VPTTA memory bank")
        count = min(int(neighbors), len(self.keys))
        query = key.detach().float().cpu().reshape(-1)
        keys = torch.stack(self.keys)
        similarities = torch.nn.functional.cosine_similarity(keys, query.unsqueeze(0), dim=1)
        scores, indices = torch.topk(similarities, k=count)
        weights = torch.softmax(scores / self.temperature, dim=0)
        prompts = torch.stack([self.prompts[int(index)] for index in indices])
        prompt = (prompts * weights.view(-1, *([1] * (prompts.ndim - 1)))).sum(dim=0)
        return prompt, float(scores.mean().item())

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "temperature": self.temperature,
            "keys": self.keys,
            "prompts": self.prompts,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("VPTTA memory capacity does not match saved state")
        if not math.isclose(float(state["temperature"]), self.temperature):
            raise ValueError("VPTTA memory temperature does not match saved state")
        self.keys = [value.detach().float().cpu().clone() for value in state["keys"]]
        self.prompts = [value.detach().float().cpu().clone() for value in state["prompts"]]
        if len(self.keys) != len(self.prompts) or len(self.keys) > self.capacity:
            raise ValueError("invalid VPTTA memory state")


class VPTTAAdapter:
    """Four-modality 3D compatibility port of VPTTA."""

    method = "vptta"

    def __init__(
        self,
        model: nn.Module,
        *,
        patch_size: Sequence[int],
        learning_rate: float = 0.05,
        betas: tuple[float, float] = (0.9, 0.99),
        prompt_alpha: float = 0.03,
        memory_size: int = 40,
        neighbors: int = 16,
        warm_n: float = 5.0,
        steps: int = 1,
    ) -> None:
        if getattr(model, "output_mode", None) != "classes_softmax":
            raise ValueError("VPTTA requires the four-class softmax source model")
        if learning_rate <= 0 or steps < 1 or neighbors < 1:
            raise ValueError("invalid VPTTA optimization settings")
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.converted_batch_norm_names = convert_to_adaptive_batch_norm_3d(
            self.model,
            warm_n=warm_n,
        )
        self.adaptive_batch_norms = [
            module for module in self.model.modules() if isinstance(module, AdaptiveBatchNorm3d)
        ]
        device = next(self.model.parameters()).device
        channels = int(getattr(self.model, "in_channels", 4))
        self.prompt = FourierPrompt3D(
            channels,
            patch_size,
            prompt_alpha=prompt_alpha,
        ).to(device)
        self.learning_rate = float(learning_rate)
        self.betas = tuple(float(value) for value in betas)
        self.prompt_alpha = float(prompt_alpha)
        self.memory_size = int(memory_size)
        self.neighbors = int(neighbors)
        self.warm_n = float(warm_n)
        self.steps = int(steps)
        self.optimizer = torch.optim.Adam(
            self.prompt.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=0.0,
        )
        self._source_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.memory = PromptMemoryBank(capacity=self.memory_size)
        self.total_updates = 0
        self.begin_case()

    def begin_case(self) -> None:
        self._case_updates = 0
        self._case_losses: list[float] = []
        self._case_memory_similarity: list[float] = []

    @torch.no_grad()
    def reset(self) -> None:
        self.prompt.reset_ones()
        self.optimizer.load_state_dict(deepcopy(self._source_optimizer_state))
        self.memory.reset()
        self.total_updates = 0
        for module in self.adaptive_batch_norms:
            module.sample_count.zero_()
            module.new_sample = False

    def _set_new_sample(self, value: bool) -> None:
        for module in self.adaptive_batch_norms:
            module.new_sample = value

    @torch.enable_grad()
    def predict_and_adapt(self, images: torch.Tensor) -> TentStepResult:
        if images.shape[0] != 1:
            raise ValueError("VPTTA follows the reference batch-size-one protocol")
        with torch.no_grad():
            key = self.prompt.low_frequency_amplitude(images)
            if len(self.memory) >= self.neighbors:
                initializer, similarity = self.memory.nearest_prompt(
                    key[0],
                    neighbors=self.neighbors,
                )
                self.prompt.update(initializer.to(self.prompt.data_prompt))
                self._case_memory_similarity.append(similarity)
            else:
                self.prompt.reset_ones()

        loss_value = float("nan")
        self.model.eval()
        self.prompt.train()
        for step_index in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            self._set_new_sample(step_index == 0)
            prompted_images, _ = self.prompt(images)
            self.model(prompted_images)
            losses = [module.bn_loss for module in self.adaptive_batch_norms]
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("VPTTA BatchNorm alignment loss became non-finite")
            loss.backward()
            self.optimizer.step()
            self._set_new_sample(False)
            loss_value = float(loss.detach().item())
            self._case_losses.append(loss_value)
            self._case_updates += 1
            self.total_updates += 1

        self.prompt.eval()
        with torch.no_grad():
            prompted_images, key = self.prompt(images)
            logits = self.model(prompted_images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            self.memory.push(key[0], self.prompt.data_prompt)
        return TentStepResult(logits=logits.detach().float(), entropy=loss_value)

    def case_diagnostics(self) -> dict[str, float | int | None]:
        return {
            "vptta_updates": self._case_updates,
            "vptta_bn_alignment_loss": (
                float(sum(self._case_losses) / len(self._case_losses))
                if self._case_losses
                else None
            ),
            "vptta_memory_size": len(self.memory),
            "vptta_memory_similarity": (
                float(sum(self._case_memory_similarity) / len(self._case_memory_similarity))
                if self._case_memory_similarity
                else None
            ),
            "vptta_prompt_mean_abs_delta": float(
                (self.prompt.data_prompt.detach() - 1.0).abs().mean().item()
            ),
            "vptta_total_updates": self.total_updates,
        }

    def adaptation_state_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt.data_prompt.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "memory": self.memory.state_dict(),
            "batch_norm_sample_counts": [
                module.sample_count.detach().cpu() for module in self.adaptive_batch_norms
            ],
            "total_updates": self.total_updates,
        }

    @torch.no_grad()
    def load_adaptation_state_dict(self, state: dict[str, Any]) -> None:
        self.prompt.update(state["prompt"].to(self.prompt.data_prompt))
        self.optimizer.load_state_dict(state["optimizer"])
        self.memory.load_state_dict(state["memory"])
        counts = state["batch_norm_sample_counts"]
        if len(counts) != len(self.adaptive_batch_norms):
            raise ValueError("VPTTA BatchNorm state does not match model")
        for module, count in zip(self.adaptive_batch_norms, counts):
            module.sample_count.copy_(count.to(module.sample_count))
            module.new_sample = False
        self.total_updates = int(state["total_updates"])

    def method_metadata(self) -> dict[str, Any]:
        return {
            "implementation": "vptta_four_modality_fft3d_v1",
            "reference_repository": "Chen-Ziyang/VPTTA@7e0a3776100d1350cd2bd0c9c3431522c5940027",
            "learning_rate": self.learning_rate,
            "optimizer": "Adam",
            "betas": list(self.betas),
            "weight_decay": 0.0,
            "prompt_alpha": self.prompt_alpha,
            "prompt_size": list(self.prompt.prompt_size),
            "prompt_channels": self.prompt.channels,
            "memory_size": self.memory_size,
            "neighbors": self.neighbors,
            "warm_n": self.warm_n,
            "steps_per_patch_batch": self.steps,
            "updated_parameters": "fourier_prompt_only",
            "exact_reference_reproduction": False,
            "compatibility_notes": [
                "2D RGB Fourier prompting is generalized to 3D four-modality prompting",
                "each sliding-window patch is treated as one continual test sample",
                "the malformed one-channel prompt fallback in the TEGDA fork is not reproduced",
            ],
        }
