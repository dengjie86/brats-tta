from __future__ import annotations

import logging
from typing import Any

import torch

from brats_tta.models import build_source_model
from brats_tta.utils.checkpoint import load_checkpoint


def configure_logging(verbose: bool = False, *, rank: int = 0) -> None:
    logging.basicConfig(
        level=(logging.DEBUG if verbose else logging.INFO) if rank == 0 else logging.WARNING,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    config_override: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    # Keep optimizer/scheduler tensors from a training checkpoint on CPU.  Only
    # the model is needed for inference, which matters on smaller local GPUs.
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    config = config_override or checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no embedded config; pass --config")
    model = build_source_model(config["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, config, checkpoint
