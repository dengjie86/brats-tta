from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from brats_tta.data.brats import BraTSDataset
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
