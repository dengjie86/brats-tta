from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from brats_tta.utils.reproducibility import resolve_device


@dataclass(frozen=True)
class DistributedContext:
    """Runtime information for a single-process or torchrun-launched job."""

    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def sum_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def close(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(requested_device: str = "auto") -> DistributedContext:
    """Initialize from torchrun environment variables, or return a single-process context."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = resolve_device(requested_device)
        return DistributedContext(False, 0, 0, 1, device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    requested = requested_device.lower()
    use_cuda = (requested == "auto" and torch.cuda.is_available()) or requested.startswith("cuda")
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed CUDA training requested, but CUDA is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible"
            )
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    elif requested == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError("distributed training supports --device auto, cuda or cpu")

    if not dist.is_available():
        raise RuntimeError("this PyTorch build does not provide torch.distributed")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(True, rank, local_rank, world_size, device, backend)


def wrap_model_for_distributed(model: nn.Module, context: DistributedContext) -> nn.Module:
    model = model.to(context.device)
    if not context.distributed:
        return model
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    output_device = context.local_rank if context.device.type == "cuda" else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model
