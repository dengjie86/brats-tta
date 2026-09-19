from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from brats_tta.cli.evaluate_tta import _validate_run_settings
from brats_tta.cli.extract_brain_masks import validate_destinations
from brats_tta.data.preprocessing import load_brain_mask, load_raw_case


def test_brain_mask_applied_before_normalization_and_never_to_label(tmp_path: Path) -> None:
    shape = (4, 4, 4)
    affine = np.eye(4)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:3] = 1
    record = {"images": {}}
    paths = []
    for i, modality in enumerate(("t1n", "t1c", "t2w", "t2f")):
        array = np.arange(1, 65, dtype=np.float32).reshape(shape) * (i + 1)
        array[mask == 0] = 10000  # Large extracranial values must not affect mean/std.
        path = tmp_path / f"{modality}.nii.gz"
        nib.save(nib.Nifti1Image(array, affine), path)
        record["images"][modality] = str(path)
        paths.append(path)
    label = np.zeros(shape, dtype=np.uint8)
    label[0, 0, 0] = 3  # Ground truth outside the predicted brain must remain evaluable.
    record["label"] = str(tmp_path / "seg.nii.gz")
    record["brain_mask"] = str(tmp_path / "mask.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), record["label"])
    nib.save(nib.Nifti1Image(mask, affine), record["brain_mask"])
    paths.extend((Path(record["label"]), Path(record["brain_mask"])))
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    image, regions, _ = load_raw_case(record, "brats_modern")
    assert np.all(image[:, mask == 0] == 0)
    for channel in image:
        assert abs(channel[mask > 0].mean()) < 1e-6
        assert abs(channel[mask > 0].std() - 1) < 1e-6
    assert regions[:, 0, 0, 0].tolist() == [1, 1, 1]
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def test_brain_mask_rejects_geometry_empty_and_nonbinary(tmp_path: Path) -> None:
    path = tmp_path / "mask.nii.gz"
    for values, affine, message in (
        (np.ones((4,4,4)), np.diag([2,1,1,1]), "affine mismatch"),
        (np.zeros((4,4,4)), np.eye(4), "empty"),
        (np.full((4,4,4), .5), np.eye(4), "binary"),
    ):
        nib.save(nib.Nifti1Image(values, affine), path)
        with pytest.raises(ValueError, match=message):
            load_brain_mask(path, (4,4,4), np.eye(4))


def test_brain_extraction_cannot_write_inside_source_tree(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    manifest = {"dataset_root": str(source), "cases": [
        {"images": {"t1c": str(source / "case/image.nii.gz")}}]}
    with pytest.raises(ValueError, match="outside"):
        validate_destinations(manifest, source / "derived", tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="outside"):
        validate_destinations(manifest, tmp_path, tmp_path / "manifest.json")
    validate_destinations(manifest, tmp_path / "outputs", tmp_path / "manifest.json")


def test_cached_scores_cannot_mix_step_counts_or_manifests(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    first = {"manifest": "raw", "steps": 1}
    _validate_run_settings(path, first, has_records=False, overwrite=False)
    _validate_run_settings(path, first, has_records=True, overwrite=False)
    for changed in ({"manifest":"brain", "steps":1}, {"manifest":"raw", "steps":10}):
        with pytest.raises(ValueError, match="settings changed"):
            _validate_run_settings(path, changed, has_records=True, overwrite=False)
    with pytest.raises(ValueError, match="lack provenance"):
        _validate_run_settings(tmp_path / "legacy.json", first, has_records=True, overwrite=False)
