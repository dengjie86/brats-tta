from __future__ import annotations

import copy
from pathlib import Path

import pytest

from brats_tta.cli.train_source import apply_train_cli_overrides, build_parser
from brats_tta.config import apply_config_overrides, load_config, validate_config
from brats_tta.models import build_source_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_config_is_valid() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "source_brats_gli.yaml")
    assert config["model"]["features"] == [32, 64, 128, 256, 320]
    assert config["model"]["track_running_stats"] is False


def test_large_four_class_source_config_is_valid() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "source_brats_gli_4class_bn.yaml")
    validate_config(config)

    assert config["model"]["features"] == [32, 64, 128, 256, 320, 320]
    assert config["model"]["out_channels"] == 4
    assert config["model"]["output_mode"] == "classes_softmax"
    assert config["model"]["norm"] == "batch3d"
    assert config["model"]["sync_batchnorm"] is True
    assert config["data"]["num_workers"] == 12
    assert config["data"]["validation_num_workers"] == 2
    assert config["data"]["prefetch_factor"] == 4
    assert config["data"]["validation_prefetch_factor"] == 1
    assert config["loss"]["deep_supervision_weights"] == [1.0, 0.5, 0.25, 0.125, 0.0]
    augmentation = config["data"]["augmentation"]
    assert augmentation["flip_axes"] == [0]
    assert augmentation["affine_probability"] == 0.5
    assert augmentation["affine_scale_range"] == [0.8, 1.2]
    assert augmentation["affine_degrees"] == 15.0
    assert augmentation["gamma_probability"] == 0.5
    assert augmentation["gamma_log_range"] == [-0.3, 0.3]
    assert augmentation["intensity_scale_probability"] == 0.0
    assert augmentation["intensity_shift_probability"] == 0.0
    assert augmentation["noise_probability"] == 0.5
    assert augmentation["noise_std_range"] == [0.0, 0.05]
    assert config["training"]["batch_size"] == 2
    assert config["training"]["epochs"] == 300
    assert config["training"]["save_every"] == 20
    assert config["training"]["amp"] is False
    assert config["inference"]["amp"] is False
    assert config["training"]["log_every"] == 1
    assert config["training"]["validation_log_every"] == 1
    assert build_source_model(config["model"]).parameter_count() == 31_199_796


def test_scale_shift_ablation_changes_only_the_intended_augmentation_probabilities() -> None:
    main = load_config(PROJECT_ROOT / "configs" / "source_brats_gli_4class_bn.yaml")
    ablation = load_config(PROJECT_ROOT / "configs" / "source_brats_gli_4class_bn_scale_shift.yaml")
    main_augmentation = main["data"]["augmentation"]
    ablation_augmentation = ablation["data"]["augmentation"]

    differing_keys = {
        key
        for key in main_augmentation
        if main_augmentation[key] != ablation_augmentation[key]
    }
    assert differing_keys == {"intensity_scale_probability", "intensity_shift_probability"}
    assert ablation_augmentation["intensity_scale_probability"] == 1.0
    assert ablation_augmentation["intensity_shift_probability"] == 1.0


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
