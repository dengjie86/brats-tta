from __future__ import annotations

import copy
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("experiment", "data", "model", "loss", "training", "inference")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = _expand_environment_variables(loaded)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def apply_config_overrides(
    config: dict[str, Any],
    overrides: Sequence[str] | None,
) -> dict[str, Any]:
    """Return a validated copy with strict dotted-key YAML overrides applied.

    Existing keys only are accepted so that a command-line typo cannot silently
    create an unused configuration option.
    """

    updated = copy.deepcopy(config)
    for override in overrides or ():
        path, separator, raw_value = override.partition("=")
        if not separator or not path.strip() or not raw_value.strip():
            raise ValueError(f"override must have the form section.key=value: {override!r}")
        keys = [key.strip() for key in path.split(".")]
        if any(not key for key in keys):
            raise ValueError(f"invalid override key: {path!r}")

        destination: dict[str, Any] = updated
        for key in keys[:-1]:
            child = destination.get(key)
            if not isinstance(child, dict):
                raise KeyError(f"unknown configuration path: {path}")
            destination = child
        final_key = keys[-1]
        if final_key not in destination:
            raise KeyError(f"unknown configuration key: {path}")
        destination[final_key] = _expand_environment_variables(yaml.safe_load(raw_value))

    validate_config(updated)
    return updated


def validate_config(config: dict[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"missing configuration sections: {', '.join(missing)}")

    model = config["model"]
    if model.get("name", "plain_unet3d") != "plain_unet3d":
        raise ValueError("only the plain_unet3d source architecture is implemented")
    if model.get("in_channels", 4) != 4:
        raise ValueError("the fixed BraTS source baseline requires four input modalities")
    if model.get("out_channels", 3) != 3:
        raise ValueError("the fixed BraTS source baseline requires ET/TC/WT output channels")
    features = model.get("features", [])
    if not isinstance(features, list) or len(features) < 2:
        raise ValueError("model.features must define at least two stages")
    if any(not isinstance(channel, int) or isinstance(channel, bool) or channel <= 0 for channel in features):
        raise ValueError("model.features must contain positive integers")
    if int(model.get("convs_per_stage", 2)) < 1:
        raise ValueError("model.convs_per_stage must be positive")
    if model.get("norm", "instance3d") != "instance3d":
        raise ValueError("PlainUNet3D currently implements InstanceNorm3d only")
    if not bool(model.get("norm_affine", True)):
        raise ValueError("the fixed source baseline requires InstanceNorm3d affine=True")
    if bool(model.get("track_running_stats", False)):
        raise ValueError("the fixed source baseline requires InstanceNorm3d track_running_stats=False")
    if model.get("activation", "leaky_relu") != "leaky_relu":
        raise ValueError("PlainUNet3D currently implements LeakyReLU only")
    if float(model.get("dropout", 0.0)) != 0.0:
        raise ValueError("the fixed source baseline does not use dropout")

    patch_size = config["data"].get("patch_size")
    if not isinstance(patch_size, list) or len(patch_size) != 3:
        raise ValueError("data.patch_size must contain D, H and W")
    divisibility = 2 ** (len(features) - 1)
    if any(int(size) % divisibility != 0 for size in patch_size):
        raise ValueError(f"data.patch_size must be divisible by {divisibility}")

    inference_patch_size = config["inference"].get("patch_size")
    if not isinstance(inference_patch_size, list) or len(inference_patch_size) != 3:
        raise ValueError("inference.patch_size must contain D, H and W")
    if any(int(size) <= 0 or int(size) % divisibility != 0 for size in inference_patch_size):
        raise ValueError(f"inference.patch_size must be positive and divisible by {divisibility}")

    weights = config["loss"].get("deep_supervision_weights")
    expected_outputs = len(features) - 1
    if model.get("deep_supervision", True) and weights is not None and len(weights) != expected_outputs:
        raise ValueError(f"expected {expected_outputs} deep-supervision weights, got {len(weights)}")
    if weights is not None and (any(float(weight) < 0 for weight in weights) or sum(weights) <= 0):
        raise ValueError("deep-supervision weights must be non-negative with a positive sum")

    training = config["training"]
    for key in ("batch_size", "epochs", "iterations_per_epoch", "validate_every", "save_every"):
        if int(training.get(key, 1)) <= 0:
            raise ValueError(f"training.{key} must be positive")


def save_config_snapshot(config: dict[str, Any], destination: str | Path) -> None:
    snapshot = copy.deepcopy(config)
    snapshot.pop("_config_path", None)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        yaml.safe_dump(snapshot, file, allow_unicode=True, sort_keys=False)


def config_fingerprint(config: dict[str, Any]) -> str:
    snapshot = copy.deepcopy(config)
    snapshot.pop("_config_path", None)
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _expand_environment_variables(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    return value
