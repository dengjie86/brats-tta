from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from brats_tta.config import save_config_snapshot
from brats_tta.engine.inference import sliding_window_logits
from brats_tta.metrics.segmentation import aggregate_metric_dicts, compute_region_metrics
from brats_tta.utils.checkpoint import load_checkpoint, save_checkpoint
from brats_tta.utils.distributed import DistributedContext, unwrap_model, wrap_model_for_distributed

LOGGER = logging.getLogger(__name__)


class SourceTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        loss_function: nn.Module,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        config: dict[str, Any],
        device: torch.device,
        distributed_context: DistributedContext | None = None,
    ) -> None:
        self.distributed = distributed_context or DistributedContext(False, 0, 0, 1, device)
        self.model = wrap_model_for_distributed(model, self.distributed)
        self.loss_function = loss_function.to(device)
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.config = config
        self.device = device
        self.training_config = config["training"]
        self.inference_config = config["inference"]

        self.output_directory = Path(config["experiment"]["output_dir"]).expanduser().resolve()
        self.checkpoint_directory = self.output_directory / "checkpoints"
        if self.distributed.is_main_process:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
            save_config_snapshot(config, self.output_directory / "config.yaml")
        self.distributed.barrier()

        self.optimizer = build_optimizer(self.model, self.training_config)
        self.scheduler = build_scheduler(self.optimizer, self.training_config)
        amp_enabled = bool(self.training_config.get("amp", True) and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.start_epoch = 0
        self.best_dice = float("-inf")
        self.history_path = self.output_directory / "history.jsonl"

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        unwrap_model(self.model).load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_dice = float(checkpoint.get("best_dice", float("-inf")))
        if self.distributed.is_main_process:
            LOGGER.info("Resumed from %s at epoch %d", checkpoint_path, self.start_epoch)
        self.distributed.barrier()

    def fit(self) -> None:
        number_of_epochs = int(self.training_config.get("epochs", 1000))
        validate_every = int(self.training_config.get("validate_every", 10))
        save_every = int(self.training_config.get("save_every", 50))
        for epoch in range(self.start_epoch, number_of_epochs):
            start_time = perf_counter()
            training_metrics = self.train_epoch(epoch)
            self.scheduler.step()
            record: dict[str, Any] = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": perf_counter() - start_time,
                **training_metrics,
            }

            should_validate = (epoch + 1) % validate_every == 0 or epoch == number_of_epochs - 1
            if should_validate and self.distributed.is_main_process:
                validation_metrics = self.validate()
                record.update({f"val_{key}": value for key, value in validation_metrics.items()})
                current_dice = validation_metrics["dice_mean"]
                if current_dice > self.best_dice:
                    self.best_dice = current_dice
                    self._save(epoch, "best.pt")

            if self.distributed.is_main_process:
                if (epoch + 1) % save_every == 0 or epoch == number_of_epochs - 1:
                    self._save(epoch, f"epoch_{epoch + 1:04d}.pt")
                self._save(epoch, "latest.pt")
                self._append_history(record)
                LOGGER.info("Epoch %d: %s", epoch + 1, _format_metrics(record))
            self.distributed.barrier()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        iterations = int(self.training_config.get("iterations_per_epoch", 250))
        gradient_clip = float(self.training_config.get("gradient_clip_norm", 12.0))
        cycle = 0
        self._set_sampler_epoch(epoch, cycle, iterations)
        iterator = iter(self.training_loader)
        running_loss = 0.0
        progress = tqdm(
            range(iterations),
            desc="source train",
            leave=False,
            disable=not self.distributed.is_main_process,
        )

        for _ in progress:
            try:
                batch = next(iterator)
            except StopIteration:
                cycle += 1
                self._set_sampler_epoch(epoch, cycle, iterations)
                iterator = iter(self.training_loader)
                batch = next(iterator)
            image = batch["image"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                outputs = self.model(image)
                loss = self.loss_function(outputs, target)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += float(loss.detach().item())
            progress.set_postfix(loss=f"{running_loss / (_ + 1):.4f}")
        totals = torch.tensor([running_loss, float(iterations)], device=self.device, dtype=torch.float64)
        self.distributed.sum_tensor(totals)
        return {"train_loss": float((totals[0] / totals[1]).item())}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if not self.distributed.is_main_process:
            raise RuntimeError("validation must only run on rank 0")
        validation_model = unwrap_model(self.model)
        validation_model.eval()
        case_metrics: list[dict[str, float]] = []
        maximum_cases = self.training_config.get("validation_cases")
        for case_index, batch in enumerate(tqdm(self.validation_loader, desc="validation", leave=False)):
            if maximum_cases is not None and case_index >= int(maximum_cases):
                break
            image = batch["image"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            if target.shape[1] == 0:
                continue
            logits = sliding_window_logits(
                validation_model,
                image,
                patch_size=self.inference_config["patch_size"],
                overlap=self.inference_config.get("overlap", 0.5),
                sw_batch_size=self.inference_config.get("sw_batch_size", 1),
                gaussian_weighting=self.inference_config.get("gaussian_weighting", True),
                amp=self.inference_config.get("amp", True),
            )
            case_metrics.append(
                compute_region_metrics(
                    logits,
                    target,
                    from_logits=True,
                    threshold=self.inference_config.get("threshold", 0.5),
                )
            )
        if not case_metrics:
            raise RuntimeError("validation manifest contains no labeled cases")
        return aggregate_metric_dicts(case_metrics)

    def _checkpoint_state(self, epoch: int) -> dict[str, Any]:
        config = copy.deepcopy(self.config)
        config.pop("_config_path", None)
        return {
            "format_version": 1,
            "epoch": epoch,
            "best_dice": self.best_dice,
            "model": unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": config,
            "distributed_world_size": self.distributed.world_size,
        }

    def _set_sampler_epoch(self, epoch: int, cycle: int, iterations: int) -> None:
        set_epoch = getattr(self.training_loader.sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch * max(iterations, 1) + cycle)

    def _save(self, epoch: int, filename: str) -> None:
        save_checkpoint(self._checkpoint_state(epoch), self.checkpoint_directory / filename)

    def _append_history(self, record: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_optimizer(model: nn.Module, training_config: dict[str, Any]) -> Optimizer:
    optimizer_name = training_config.get("optimizer", "sgd").lower()
    learning_rate = float(training_config.get("learning_rate", 1e-2))
    weight_decay = float(training_config.get("weight_decay", 3e-5))
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(training_config.get("momentum", 0.99)),
            nesterov=bool(training_config.get("nesterov", True)),
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer: Optimizer, training_config: dict[str, Any]) -> LambdaLR:
    number_of_epochs = int(training_config.get("epochs", 1000))
    exponent = float(training_config.get("poly_exponent", 0.9))

    def polynomial_decay(epoch: int) -> float:
        progress = min(max(epoch, 0), number_of_epochs) / max(number_of_epochs, 1)
        return (1.0 - progress) ** exponent

    return LambdaLR(optimizer, lr_lambda=polynomial_decay)


def _format_metrics(metrics: dict[str, Any]) -> str:
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}={value:.4f}")
    return ", ".join(formatted)
