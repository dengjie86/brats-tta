from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from brats_tta.data.manifest import load_manifest
from brats_tta.data.preprocessing import load_raw_case
from brats_tta.data.transforms import SourcePatchTransform
from brats_tta.utils.distributed import DistributedContext


class BraTSDataset(Dataset[dict[str, Any]]):
    """BraTS dataset supporting both raw NIfTI and preprocessed memory-mapped manifests."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        training: bool,
        patch_size: tuple[int, int, int] | None = None,
        label_schema: str = "brats_modern",
        augmentation_config: dict[str, Any] | None = None,
    ) -> None:
        self.manifest = load_manifest(manifest_path)
        self.cases = self.manifest["cases"]
        self.training = bool(training)
        self.label_schema = self.manifest.get("label_schema", label_schema)
        self.transform: SourcePatchTransform | None = None
        if self.training:
            if patch_size is None:
                raise ValueError("training requires a patch size")
            augmentation_config = augmentation_config or {}
            self.transform = SourcePatchTransform(
                patch_size,
                foreground_oversample=augmentation_config.get("foreground_oversample", 0.33),
                flip_probability=augmentation_config.get("flip_probability", 0.5),
                intensity_scale_range=tuple(augmentation_config.get("intensity_scale_range", [0.9, 1.1])),
                intensity_shift_range=tuple(augmentation_config.get("intensity_shift_range", [-0.1, 0.1])),
                noise_probability=augmentation_config.get("noise_probability", 0.15),
                noise_std_range=tuple(augmentation_config.get("noise_std_range", [0.0, 0.1])),
            )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.cases[index]
        image, target = self._load_record(record)
        if target is None:
            if self.training:
                raise ValueError(f"training case {record['id']} has no label")
            target = torch.empty((0, *image.shape[1:]), dtype=torch.float32)
        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            image = image.clone().contiguous()
            target = target.clone().contiguous()
        return {
            "id": record["id"],
            "image": image.float(),
            "target": target.float(),
            "reference": record.get("reference") or record.get("images", {}).get("t1n"),
            # default_collate cannot batch None, which is expected for target-domain inference cases.
            "label": record.get("label") or "",
        }

    def _load_record(self, record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if "image" in record:
            # Copy-on-write mmap is writable from PyTorch's point of view but never mutates the cached .npy.
            image_array = np.load(record["image"], mmap_mode="c", allow_pickle=False)
            image = torch.from_numpy(np.asarray(image_array))
            target: torch.Tensor | None = None
            if record.get("regions"):
                region_array = np.load(record["regions"], mmap_mode="c", allow_pickle=False)
                target = torch.from_numpy(np.asarray(region_array))
            return image, target

        images, regions, _ = load_raw_case(record, self.label_schema)
        image = torch.from_numpy(images)
        target = torch.from_numpy(regions) if regions is not None else None
        return image, target


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation cases across ranks without padding or duplication."""

    def __init__(self, dataset: Dataset[Any], *, num_replicas: int, rank: int) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas


def build_dataloaders(
    config: dict[str, Any],
    distributed_context: DistributedContext | None = None,
) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    training_config = config["training"]
    patch_size = tuple(data_config["patch_size"])
    label_schema = data_config.get("label_schema", "brats_modern")
    training_dataset = BraTSDataset(
        data_config["train_manifest"],
        training=True,
        patch_size=patch_size,
        label_schema=label_schema,
        augmentation_config=data_config.get("augmentation", {}),
    )
    validation_dataset = BraTSDataset(
        data_config["val_manifest"],
        training=False,
        label_schema=label_schema,
    )

    workers = int(data_config.get("num_workers", 4))
    seed = int(config["experiment"].get("seed", 2025))
    rank = distributed_context.rank if distributed_context is not None else 0
    world_size = distributed_context.world_size if distributed_context is not None else 1
    is_distributed = bool(distributed_context is not None and distributed_context.distributed)
    training_generator = torch.Generator().manual_seed(seed + rank * 100_003)
    validation_generator = torch.Generator().manual_seed(seed + 1 + rank * 100_003)
    common = {
        "num_workers": workers,
        "pin_memory": bool(data_config.get("pin_memory", True) and torch.cuda.is_available()),
        "persistent_workers": workers > 0,
        "worker_init_fn": _seed_worker,
    }
    batch_size = int(training_config.get("batch_size", 2))
    drop_last = len(training_dataset) >= batch_size * world_size
    training_sampler = (
        DistributedSampler(
            training_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=drop_last,
        )
        if is_distributed
        else None
    )
    validation_sampler = (
        DistributedEvalSampler(validation_dataset, num_replicas=world_size, rank=rank)
        if is_distributed
        else None
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=training_sampler is None,
        sampler=training_sampler,
        drop_last=drop_last,
        generator=training_generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        sampler=validation_sampler,
        drop_last=False,
        generator=validation_generator,
        **common,
    )
    return training_loader, validation_loader


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
