from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
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


def _progress_disabled(distributed: DistributedContext) -> bool:
    """Avoid terminal progress writes when training is detached or piped."""
    if not distributed.is_main_process:
        return True
    value = os.environ.get("TQDM_DISABLE", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class _CudaBatchPrefetcher:
    """Copy the next pinned batch while the current batch is executing."""

    def __init__(self, iterator: Any, device: torch.device) -> None:
        self.iterator = iterator
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_image: torch.Tensor | None = None
        self.next_target: torch.Tensor | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_image = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_image = batch["image"].to(self.device, non_blocking=True)
            self.next_target = batch["target"].to(self.device, non_blocking=True)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.next_image is None or self.next_target is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)
        image = self.next_image
        target = self.next_target
        image.record_stream(current_stream)
        target.record_stream(current_stream)
        self._preload()
        return image, target


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
        self.sync_batchnorm = bool(config["model"].get("sync_batchnorm", False))
        self.model = wrap_model_for_distributed(
            model,
            self.distributed,
            sync_batchnorm=self.sync_batchnorm,
        )
        self.loss_function = loss_function.to(device)
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.config = config
        self.device = device
        self.training_config = config["training"]
        self.inference_config = config["inference"]
        self.output_mode = config["model"].get("output_mode", "regions_sigmoid")
        self.label_schema = config["data"].get("label_schema", "brats_modern")

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
            epoch_started_at = datetime.now(timezone.utc).isoformat()
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            if self.distributed.is_main_process:
                LOGGER.info("Epoch %d/%d started", epoch + 1, number_of_epochs)
            start_time = perf_counter()
            training_metrics = self.train_epoch(epoch)
            self.scheduler.step()
            record: dict[str, Any] = {
                "epoch": epoch,
                "completed_epoch": epoch + 1,
                "global_step": (epoch + 1) * int(self.training_config.get("iterations_per_epoch", 250)),
                "started_at_utc": epoch_started_at,
                "lr": learning_rate,
                "next_lr": float(self.optimizer.param_groups[0]["lr"]),
                "train_seconds": perf_counter() - start_time,
                **training_metrics,
            }
            saved_checkpoints: list[str] = []

            should_validate = (epoch + 1) % validate_every == 0 or epoch == number_of_epochs - 1
            if should_validate:
                validation_start = perf_counter()
                if self.distributed.is_main_process:
                    LOGGER.info(
                        "Epoch %d validation started across %d process(es)",
                        epoch + 1,
                        self.distributed.world_size,
                    )
                validation_metrics = self.validate()
                if self.distributed.is_main_process:
                    record["val_seconds"] = perf_counter() - validation_start
                    record.update({f"val_{key}": value for key, value in validation_metrics.items()})
                    current_dice = validation_metrics["dice_mean"]
                    if current_dice > self.best_dice:
                        self.best_dice = current_dice
                        self._save(epoch, "best.pt")
                        saved_checkpoints.append("best.pt")
                        LOGGER.info(
                            "New best validation Dice %.6f at epoch %d",
                            self.best_dice,
                            epoch + 1,
                        )

            if self.distributed.is_main_process:
                if (epoch + 1) % save_every == 0:
                    self._save(epoch, f"epoch_{epoch + 1:04d}.pt")
                    saved_checkpoints.append(f"epoch_{epoch + 1:04d}.pt")
                if epoch == number_of_epochs - 1:
                    self._save(epoch, "last.pt")
                    saved_checkpoints.append("last.pt")
                self._save(epoch, "latest.pt")
                saved_checkpoints.append("latest.pt")
                record["best_dice"] = self.best_dice
                record["checkpoints"] = saved_checkpoints
                record["seconds"] = perf_counter() - start_time
                record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                self._append_history(record)
                LOGGER.info("Epoch %d: %s", epoch + 1, _format_metrics(record))
            self.distributed.barrier()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        iterations = int(self.training_config.get("iterations_per_epoch", 250))
        gradient_clip = float(self.training_config.get("gradient_clip_norm", 12.0))
        cycle = 0
        self._set_sampler_epoch(epoch, cycle, iterations)
        iterator = iter(self.training_loader)
        prefetcher = (
            _CudaBatchPrefetcher(iterator, self.device) if self.device.type == "cuda" else None
        )
        running_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        log_every = int(self.training_config.get("log_every", 10))
        progress = tqdm(
            range(iterations),
            desc="source train",
            leave=False,
            disable=_progress_disabled(self.distributed),
        )

        for _ in progress:
            try:
                if prefetcher is not None:
                    image, target = prefetcher.next()
                else:
                    batch = next(iterator)
                    image = batch["image"].to(self.device, non_blocking=True)
                    target = batch["target"].to(self.device, non_blocking=True)
            except StopIteration:
                cycle += 1
                self._set_sampler_epoch(epoch, cycle, iterations)
                iterator = iter(self.training_loader)
                prefetcher = (
                    _CudaBatchPrefetcher(iterator, self.device) if self.device.type == "cuda" else None
                )
                if prefetcher is not None:
                    image, target = prefetcher.next()
                else:
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
            running_loss.add_(loss.detach().to(dtype=running_loss.dtype))
            if self.distributed.is_main_process and ((_ + 1) % log_every == 0 or _ + 1 == iterations):
                current_loss = float(loss.detach().item())
                running_loss_value = float(running_loss.div(_ + 1).item())
                progress.set_postfix(loss=f"{running_loss_value:.4f}")
                LOGGER.info(
                    "Epoch %d train iteration %d/%d: rank0_loss=%.6f, rank0_running_loss=%.6f, lr=%.8g",
                    epoch + 1,
                    _ + 1,
                    iterations,
                    current_loss,
                    running_loss_value,
                    self.optimizer.param_groups[0]["lr"],
                )
        totals = torch.stack(
            (
                running_loss,
                torch.tensor(float(iterations), device=self.device, dtype=torch.float64),
            )
        )
        self.distributed.sum_tensor(totals)
        metrics = {"train_loss": float((totals[0] / totals[1]).item())}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(self.device) / (1024**2),
                    torch.cuda.max_memory_reserved(self.device) / (1024**2),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            self.distributed.max_tensor(peak_memory)
            metrics["gpu_peak_allocated_mb"] = float(peak_memory[0].item())
            metrics["gpu_peak_reserved_mb"] = float(peak_memory[1].item())
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        validation_model = unwrap_model(self.model)
        validation_model.eval()
        case_records: list[dict[str, Any]] = []
        maximum_cases = self.training_config.get("validation_cases")
        validation_log_every = int(self.training_config.get("validation_log_every", 10))
        progress = tqdm(
            self.validation_loader,
            desc="validation",
            leave=False,
            disable=_progress_disabled(self.distributed),
        )
        for local_case_index, batch in enumerate(progress):
            global_case_index = (
                self.distributed.rank + local_case_index * self.distributed.world_size
                if self.distributed.distributed
                else local_case_index
            )
            if maximum_cases is not None and global_case_index >= int(maximum_cases):
                break
            image = batch["image"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            if target.numel() == 0:
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
            metrics = compute_region_metrics(
                logits,
                target,
                from_logits=True,
                threshold=self.inference_config.get("threshold", 0.5),
                output_mode=self.output_mode,
                label_schema=self.label_schema,
            )
            case_records.append(
                {
                    "case_index": global_case_index,
                    "case_id": str(batch["id"][0]),
                    **metrics,
                }
            )
        gathered_records = self.distributed.all_gather_objects(case_records)
        case_records = [record for rank_records in gathered_records for record in rank_records]
        case_records.sort(key=lambda record: int(record["case_index"]))
        if self.distributed.is_main_process:
            for case_index, case_record in enumerate(case_records, start=1):
                if case_index % validation_log_every == 0 or case_index == len(case_records):
                    LOGGER.info(
                        "Validation case %d/%d id=%s: %s",
                        case_index,
                        len(case_records),
                        case_record["case_id"],
                        _format_metrics(case_record),
                    )
        case_metrics = [
            {key: value for key, value in record.items() if key not in {"case_id", "case_index"}}
            for record in case_records
        ]
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
        destination = self.checkpoint_directory / filename
        save_checkpoint(self._checkpoint_state(epoch), destination)
        LOGGER.info(
            "Saved checkpoint %s at completed epoch %d (%d bytes)",
            destination,
            epoch + 1,
            destination.stat().st_size,
        )

    def _append_history(self, record: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())


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
