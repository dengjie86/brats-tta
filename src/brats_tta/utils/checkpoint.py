from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"not a valid project checkpoint: {checkpoint_path}")
    return checkpoint
