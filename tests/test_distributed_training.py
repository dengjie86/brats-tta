from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from brats_tta.config import save_config_snapshot
from brats_tta.data.manifest import write_manifest
from brats_tta.models import PlainUNet3D
from brats_tta.utils.checkpoint import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_preprocessed_case(root: Path, case_id: str) -> dict[str, Any]:
    case_directory = root / case_id
    case_directory.mkdir(parents=True)
    generator = np.random.default_rng(int(case_id.rsplit("-", 1)[1]))
    image = generator.normal(size=(4, 8, 8, 8)).astype(np.float32)
    regions = np.zeros((3, 8, 8, 8), dtype=np.uint8)
    regions[2, 1:7, 1:7, 1:7] = 1
    regions[1, 2:6, 2:6, 2:6] = 1
    regions[0, 3:5, 3:5, 3:5] = 1
    image_path = case_directory / "images.npy"
    region_path = case_directory / "regions.npy"
    np.save(image_path, image, allow_pickle=False)
    np.save(region_path, regions, allow_pickle=False)
    return {
        "id": case_id,
        "image": str(image_path),
        "regions": str(region_path),
        "reference": "synthetic.nii.gz",
        "label": "synthetic-seg.nii.gz",
    }


def _distributed_config(tmp_path: Path) -> tuple[Path, Path]:
    cases = [_write_preprocessed_case(tmp_path / "data", f"Case-{index}") for index in range(4)]
    train_manifest = tmp_path / "train.json"
    val_manifest = tmp_path / "val.json"
    write_manifest(cases, train_manifest, metadata={"preprocessed": True})
    write_manifest(cases[:1], val_manifest, metadata={"preprocessed": True})
    output_directory = tmp_path / "run"
    config = {
        "experiment": {
            "name": "ddp_smoke",
            "output_dir": str(output_directory),
            "seed": 7,
            "deterministic": False,
        },
        "data": {
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "label_schema": "brats_modern",
            "patch_size": [8, 8, 8],
            "num_workers": 0,
            "pin_memory": False,
            "augmentation": {
                "foreground_oversample": 0.0,
                "flip_probability": 0.0,
                "intensity_scale_range": [1.0, 1.0],
                "intensity_shift_range": [0.0, 0.0],
                "noise_probability": 0.0,
                "noise_std_range": [0.0, 0.0],
            },
        },
        "model": {
            "name": "plain_unet3d",
            "in_channels": 4,
            "out_channels": 3,
            "features": [2, 4, 8],
            "convs_per_stage": 2,
            "deep_supervision": True,
            "norm": "instance3d",
            "norm_affine": True,
            "track_running_stats": False,
            "activation": "leaky_relu",
            "dropout": 0.0,
        },
        "loss": {
            "dice_weight": 1.0,
            "bce_weight": 1.0,
            "deep_supervision_weights": [1.0, 0.0],
        },
        "training": {
            "device": "cpu",
            "batch_size": 1,
            "epochs": 1,
            "iterations_per_epoch": 1,
            "validate_every": 1,
            "save_every": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 0.0,
            "poly_exponent": 0.9,
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
    config_path = tmp_path / "ddp.yaml"
    save_config_snapshot(config, config_path)
    return config_path, output_directory


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_two_process_ddp_train_validate_and_checkpoint(tmp_path: Path) -> None:
    config_path, output_directory = _distributed_config(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["OMP_NUM_THREADS"] = "1"
    environment["USE_LIBUV"] = "0"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous_socket:
        rendezvous_socket.bind(("127.0.0.1", 0))
        rendezvous_port = rendezvous_socket.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "brats_tta.cli.train_source",
        "--config",
        str(config_path),
    ]
    processes: list[subprocess.Popen[str]] = []
    for rank in range(2):
        rank_environment = environment.copy()
        rank_environment.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(rendezvous_port),
                "WORLD_SIZE": "2",
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
            }
        )
        processes.append(
            subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=rank_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    outputs: list[str] = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=120)
            outputs.append(stdout + "\n" + stderr)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
    assert all(process.returncode == 0 for process in processes), "\n".join(outputs)

    checkpoint = load_checkpoint(output_directory / "checkpoints" / "latest.pt")
    assert checkpoint["distributed_world_size"] == 2
    assert not any(key.startswith("module.") for key in checkpoint["model"])
    PlainUNet3D(features=(2, 4, 8)).load_state_dict(checkpoint["model"], strict=True)
    history_lines = (output_directory / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1
    history_record = json.loads(history_lines[0])
    assert history_record["epoch"] == 0
    assert "val_dice_mean" in history_record
    assert "val_seconds" in history_record
