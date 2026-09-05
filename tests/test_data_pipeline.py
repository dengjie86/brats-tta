from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from brats_tta.data.brats import BraTSDataset, DistributedEvalSampler
from brats_tta.data.manifest import discover_brats_cases, load_manifest, split_cases, write_manifest
from brats_tta.data.preprocessing import preprocess_manifest


def _write_case(root: Path, case_id: str, *, with_label: bool = True) -> None:
    case_directory = root / case_id
    case_directory.mkdir(parents=True)
    shape = (9, 10, 11)
    affine = np.diag([1.0, 1.2, 1.5, 1.0])
    generator = np.random.default_rng(abs(hash(case_id)) % (2**32))
    for modality_index, modality in enumerate(("t1n", "t1c", "t2w", "t2f")):
        image = np.zeros(shape, dtype=np.float32)
        image[1:-1, 1:-1, 1:-1] = generator.normal(
            loc=float(modality_index),
            scale=1.0,
            size=(7, 8, 9),
        )
        nib.save(nib.Nifti1Image(image, affine), case_directory / f"{case_id}-{modality}.nii.gz")
    if with_label:
        label = np.zeros(shape, dtype=np.uint8)
        label[2:7, 2:8, 2:9] = 2
        label[3:6, 3:7, 3:8] = 1
        label[4:5, 4:6, 4:7] = 3
        nib.save(nib.Nifti1Image(label, affine), case_directory / f"{case_id}-seg.nii.gz")


def _write_nested_legacy_case(root: Path, case_id: str) -> None:
    case_directory = root / case_id
    shape = (9, 10, 11)
    affine = np.eye(4)
    legacy_names = {
        "t1": 1.0,
        "t1ce": 2.0,
        "t2": 3.0,
        "flair": 4.0,
    }
    for legacy_name, value in legacy_names.items():
        modality_directory = case_directory / f"{case_id}_{legacy_name}.nii"
        modality_directory.mkdir(parents=True)
        image = np.zeros(shape, dtype=np.float32)
        image[1:-1, 1:-1, 1:-1] = value
        nib.save(
            nib.Nifti1Image(image, affine),
            modality_directory / f"00000057_brain_{legacy_name}.nii",
        )

    label_directory = case_directory / f"{case_id}_seg.nii"
    label_directory.mkdir(parents=True)
    label = np.zeros(shape, dtype=np.uint8)
    label[2:7, 2:8, 2:9] = 2
    label[3:6, 3:7, 3:8] = 1
    label[4:5, 4:6, 4:7] = 4
    # This Kaggle mirror uses the revised *_seg_new.nii name for 235 cases.
    nib.save(nib.Nifti1Image(label, affine), label_directory / f"{case_id}_seg_new.nii")


def test_manifest_preprocess_and_dataset(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_case(raw_root, "Case-001")
    _write_case(raw_root, "Case-002")
    cases = discover_brats_cases(raw_root)
    assert [case["id"] for case in cases] == ["Case-001", "Case-002"]

    training, validation = split_cases(cases, validation_fraction=0.5, seed=17)
    assert len(training) == len(validation) == 1
    assert {training[0]["id"], validation[0]["id"]} == {"Case-001", "Case-002"}

    raw_manifest = tmp_path / "raw.json"
    write_manifest(cases, raw_manifest)
    processed_manifest = tmp_path / "processed.json"
    preprocess_manifest(
        raw_manifest,
        tmp_path / "preprocessed",
        processed_manifest,
        label_schema="brats_modern",
    )
    manifest = load_manifest(processed_manifest)
    assert manifest["preprocessed"] is True
    assert manifest["label_schema"] == "brats_modern"

    dataset = BraTSDataset(
        processed_manifest,
        training=True,
        patch_size=(12, 12, 12),
        augmentation_config={
            "foreground_oversample": 1.0,
            "flip_probability": 0.0,
            "intensity_scale_range": [1.0, 1.0],
            "intensity_shift_range": [0.0, 0.0],
            "noise_probability": 0.0,
        },
    )
    sample = dataset[0]
    assert sample["image"].shape == (4, 12, 12, 12)
    assert sample["target"].shape == (3, 12, 12, 12)
    assert torch.all(sample["target"][0] <= sample["target"][1])
    assert torch.all(sample["target"][1] <= sample["target"][2])


def test_unlabeled_target_case_can_be_collated(tmp_path: Path) -> None:
    raw_root = tmp_path / "target"
    _write_case(raw_root, "Target-001", with_label=False)
    cases = discover_brats_cases(raw_root, require_label=False)
    manifest_path = tmp_path / "target.json"
    write_manifest(cases, manifest_path)

    loader = DataLoader(BraTSDataset(manifest_path, training=False), batch_size=1)
    batch = next(iter(loader))
    assert batch["label"] == [""]
    assert batch["target"].shape == (1, 0, 9, 10, 11)


def test_manifest_can_skip_incomplete_cases_without_modifying_source(tmp_path: Path) -> None:
    raw_root = tmp_path / "target"
    _write_case(raw_root, "Complete-001")
    _write_case(raw_root, "Incomplete-002")
    (raw_root / "Incomplete-002" / "Incomplete-002-t1c.nii.gz").unlink()

    with pytest.raises(ValueError, match="missing modalities"):
        discover_brats_cases(raw_root)

    cases = discover_brats_cases(raw_root, skip_incomplete=True)
    assert [case["id"] for case in cases] == ["Complete-001"]


def test_nested_kaggle_brats2021_layout(tmp_path: Path) -> None:
    raw_root = tmp_path / "BRaTS 2021 Task 1 Dataset"
    _write_nested_legacy_case(raw_root, "BraTS2021_00000")

    cases = discover_brats_cases(raw_root)
    assert len(cases) == 1
    assert cases[0]["id"] == "BraTS2021_00000"
    assert Path(cases[0]["images"]["t1n"]).name.endswith("_brain_t1.nii")
    assert Path(cases[0]["images"]["t1c"]).name.endswith("_brain_t1ce.nii")
    assert Path(cases[0]["images"]["t2w"]).name.endswith("_brain_t2.nii")
    assert Path(cases[0]["images"]["t2f"]).name.endswith("_brain_flair.nii")

    manifest_path = tmp_path / "nested.json"
    write_manifest(cases, manifest_path, metadata={"label_schema": "brats_legacy"})
    sample = BraTSDataset(manifest_path, training=False)[0]
    assert sample["image"].shape == (4, 9, 10, 11)
    assert sample["target"].shape == (3, 9, 10, 11)
    assert torch.all(sample["target"][0] <= sample["target"][1])
    assert torch.all(sample["target"][1] <= sample["target"][2])


def test_distributed_eval_sampler_has_no_padding_or_duplicates() -> None:
    dataset = list(range(7))
    shards = [
        list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank))
        for rank in range(3)
    ]

    assert shards == [[0, 3, 6], [1, 4], [2, 5]]
    assert sorted(index for shard in shards for index in shard) == list(range(7))
