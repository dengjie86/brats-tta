from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from brats_tta.engine.trainer import SourceTrainer
from brats_tta.losses.segmentation import build_loss
from brats_tta.models.unet3d import PlainUNet3D


class SingleCaseDataset(Dataset[dict[str, Any]]):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(12)
        self.image = torch.randn((4, 8, 8, 8), generator=generator)
        self.target = torch.zeros((3, 8, 8, 8))
        self.target[2, 1:7, 1:7, 1:7] = 1
        self.target[1, 2:6, 2:6, 2:6] = 1
        self.target[0, 3:5, 3:5, 3:5] = 1

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        assert index == 0
        return {
            "id": "synthetic",
            "image": self.image.clone(),
            "target": self.target.clone(),
            "reference": "synthetic.nii.gz",
            "label": "synthetic-seg.nii.gz",
        }


def _config(output_directory: Path) -> dict[str, Any]:
    return {
        "experiment": {"output_dir": str(output_directory), "seed": 12},
        "data": {"patch_size": [8, 8, 8]},
        "model": {
            "in_channels": 4,
            "out_channels": 3,
            "features": [2, 4, 8],
            "deep_supervision": True,
        },
        "loss": {"deep_supervision_weights": [1.0, 0.0]},
        "training": {
            "optimizer": "adamw",
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "batch_size": 1,
            "epochs": 1,
            "iterations_per_epoch": 1,
            "validate_every": 1,
            "save_every": 1,
            "gradient_clip_norm": 1.0,
            "amp": False,
        },
        "inference": {
            "patch_size": [8, 8, 8],
            "overlap": 0.5,
            "sw_batch_size": 1,
            "gaussian_weighting": True,
            "amp": False,
            "threshold": 0.5,
        },
    }


def _build_trainer(output_directory: Path) -> SourceTrainer:
    model = PlainUNet3D(features=(2, 4, 8), deep_supervision=True)
    loss = build_loss({"deep_supervision_weights": [1.0, 0.0]}, number_of_outputs=2)
    loader = DataLoader(SingleCaseDataset(), batch_size=1)
    return SourceTrainer(
        model=model,
        loss_function=loss,
        training_loader=loader,
        validation_loader=loader,
        config=_config(output_directory),
        device=torch.device("cpu"),
    )


def test_complete_train_validate_checkpoint_and_resume(tmp_path: Path) -> None:
    output_directory = tmp_path / "run"
    trainer = _build_trainer(output_directory)
    trainer.fit()

    latest = output_directory / "checkpoints" / "latest.pt"
    assert latest.exists()
    assert (output_directory / "checkpoints" / "best.pt").exists()
    assert (output_directory / "checkpoints" / "epoch_0001.pt").exists()
    with (output_directory / "history.jsonl").open(encoding="utf-8") as file:
        record = json.loads(file.readline())
    assert record["epoch"] == 0
    assert "train_loss" in record and "val_dice_mean" in record

    resumed = _build_trainer(output_directory)
    resumed.resume(latest)
    assert resumed.start_epoch == 1
    assert resumed.best_dice == trainer.best_dice
