from __future__ import annotations

import copy
from pathlib import Path

import pytest

from brats_tta.cli.train_source import apply_train_cli_overrides, build_parser
from brats_tta.config import apply_config_overrides, load_config, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_config_is_valid() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "source_brats_gli.yaml")
    assert config["model"]["features"] == [32, 64, 128, 256, 320]
    assert config["model"]["track_running_stats"] is False


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("model", "track_running_stats", True, "track_running_stats=False"),
        ("model", "dropout", 0.1, "does not use dropout"),
        ("inference", "patch_size", [127, 128, 128], "divisible by 16"),
    ],
)
def test_unsupported_source_settings_fail_fast(
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "source_brats_gli.yaml")
    config = copy.deepcopy(config)
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_training_command_line_overrides() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "source_brats_gli.yaml")
    args = build_parser().parse_args(
        [
            "--config",
            "base.yaml",
            "--train-manifest",
            "/kaggle/input/brats/gli_train.json",
            "--val-manifest",
            "/kaggle/input/brats/gli_val.json",
            "--output-dir",
            "/kaggle/working/source_run",
            "--batch-size",
            "1",
            "--epochs",
            "20",
            "--patch-size",
            "112",
            "112",
            "112",
            "--amp",
            "--set",
            "data.augmentation.noise_probability=0.0",
            "--set",
            "training.validation_cases=3",
        ]
    )
    updated = apply_train_cli_overrides(config, args)

    assert updated["data"]["train_manifest"] == "/kaggle/input/brats/gli_train.json"
    assert updated["data"]["val_manifest"] == "/kaggle/input/brats/gli_val.json"
    assert updated["experiment"]["output_dir"] == "/kaggle/working/source_run"
    assert updated["training"]["batch_size"] == 1
    assert updated["training"]["epochs"] == 20
    assert updated["training"]["amp"] is True
    assert updated["data"]["patch_size"] == [112, 112, 112]
    assert updated["inference"]["patch_size"] == [112, 112, 112]
    assert updated["data"]["augmentation"]["noise_probability"] == 0.0
    assert updated["training"]["validation_cases"] == 3


def test_generic_override_rejects_unknown_keys() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "source_brats_gli.yaml")
    with pytest.raises(KeyError, match="unknown configuration key"):
        apply_config_overrides(config, ["training.learnng_rate=0.1"])
