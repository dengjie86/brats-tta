"""Atomic patient-boundary state for continual TTA baselines."""
from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _atomic_torch_save(path: Path, state: dict[str, Any], *, attempts: int = 20) -> None:
    """Stream a potentially large state to a sibling file before replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(min(0.05 * 2**attempt, 0.5))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except PermissionError:
                pass


@contextmanager
def source_parameters(adapter):
    """Temporarily expose source parameters without disturbing optimizer state."""

    current = [parameter.detach().clone() for parameter in adapter.parameters]
    try:
        with torch.no_grad():
            for parameter, source in zip(adapter.parameters, adapter._source_parameters):
                parameter.copy_(source)
        yield adapter.model
    finally:
        with torch.no_grad():
            for parameter, saved in zip(adapter.parameters, current):
                parameter.copy_(saved)


def save_baseline_state(path: Path, adapter, settings: dict[str, Any], records: list[dict]) -> None:
    state = {
        "settings": settings,
        "records": records,
        "names": adapter.parameter_names,
        "parameters": [parameter.detach().cpu().clone() for parameter in adapter.parameters],
        "optimizer": _cpu_clone(adapter.optimizer.state_dict()),
        "scaler": adapter.scaler.state_dict(),
        "adapter": _cpu_clone(adapter.continual_state_dict()),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    _atomic_torch_save(Path(path), state)


def load_baseline_state(path: Path, adapter, settings: dict[str, Any]) -> list[dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["settings"] != settings or state["names"] != adapter.parameter_names:
        raise ValueError("Continual checkpoint protocol or parameter names mismatch")
    if len(state["parameters"]) != len(adapter.parameters):
        raise ValueError("Continual checkpoint parameter count mismatch")
    with torch.no_grad():
        for parameter, saved in zip(adapter.parameters, state["parameters"]):
            if parameter.shape != saved.shape:
                raise ValueError("Continual checkpoint parameter shape mismatch")
            parameter.copy_(saved)
    adapter.optimizer.load_state_dict(state["optimizer"])
    adapter.scaler.load_state_dict(state["scaler"])
    adapter.load_continual_state_dict(state["adapter"])
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["records"]
