from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from brats_tta.data.manifest import MODALITY_ORDER, load_manifest, write_manifest

LABEL_SCHEMAS: dict[str, dict[str, tuple[int, ...] | int]] = {
    "brats_modern": {
        "et": (3,),
        "tc": (1, 3),
        "wt": (1, 2, 3),
        "et_value": 3,
        "tc_value": 1,
        "wt_value": 2,
    },
    "brats_legacy": {
        "et": (4,),
        "tc": (1, 4),
        "wt": (1, 2, 4),
        "et_value": 4,
        "tc_value": 1,
        "wt_value": 2,
    },
    # BraTS-PEDs 2024 provides four mutually exclusive tissues:
    # 1=ET, 2=NET, 3=CC, 4=ED.  The adult source model predicts the
    # three nested evaluation regions, so NET and CC are merged into TC.
    "brats_ped_2024": {
        "et": (1,),
        "tc": (1, 2, 3),
        "wt": (1, 2, 3, 4),
        "et_value": 1,
        "tc_value": 2,
        "wt_value": 4,
    },
}


def zscore_nonzero(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    mask = image != 0
    output = np.zeros_like(image, dtype=np.float32)
    if not np.any(mask):
        return output
    values = image[mask]
    standard_deviation = float(values.std())
    if standard_deviation < eps:
        output[mask] = values - float(values.mean())
    else:
        output[mask] = (values - float(values.mean())) / standard_deviation
    return output


def labelmap_to_regions(label: np.ndarray, schema: str) -> np.ndarray:
    if schema not in LABEL_SCHEMAS:
        raise ValueError(f"unknown label schema {schema!r}; choose from {sorted(LABEL_SCHEMAS)}")
    mapping = LABEL_SCHEMAS[schema]
    label = np.asarray(label)
    allowed_values = {0, *mapping["wt"]}
    observed_values = set(int(value) for value in np.unique(label))
    unexpected = observed_values - allowed_values
    if unexpected:
        raise ValueError(f"label contains values not defined by {schema}: {sorted(unexpected)}")
    return np.stack(
        (
            np.isin(label, mapping["et"]),
            np.isin(label, mapping["tc"]),
            np.isin(label, mapping["wt"]),
        ),
        axis=0,
    ).astype(np.uint8)


def regions_to_labelmap(
    regions: np.ndarray,
    schema: str,
    *,
    threshold: float = 0.5,
    enforce_hierarchy: bool = True,
) -> np.ndarray:
    if schema not in LABEL_SCHEMAS:
        raise ValueError(f"unknown label schema {schema!r}")
    if regions.shape[0] != 3:
        raise ValueError("regions must have channels ordered ET, TC, WT")
    et, tc, wt = np.asarray(regions) >= threshold
    if enforce_hierarchy:
        tc = tc | et
        wt = wt | tc
    result = np.zeros(regions.shape[1:], dtype=np.uint8)
    result[wt] = int(LABEL_SCHEMAS[schema]["wt_value"])
    result[tc] = int(LABEL_SCHEMAS[schema]["tc_value"])
    result[et] = int(LABEL_SCHEMAS[schema]["et_value"])
    return result


def load_raw_case(record: dict[str, Any], label_schema: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    images: list[np.ndarray] = []
    reference_image: nib.spatialimages.SpatialImage | None = None
    reference_shape: tuple[int, ...] | None = None
    reference_affine: np.ndarray | None = None

    for modality in MODALITY_ORDER:
        image_path = record["images"][modality]
        image_object = nib.load(image_path)
        if reference_image is None:
            reference_image = image_object
            reference_shape = image_object.shape
            reference_affine = image_object.affine
        _validate_geometry(image_object, image_path, reference_shape, reference_affine)
        images.append(zscore_nonzero(image_object.get_fdata(dtype=np.float32)))

    stacked_images = np.stack(images, axis=0).astype(np.float32, copy=False)
    regions: np.ndarray | None = None
    if record.get("label"):
        label_object = nib.load(record["label"])
        _validate_geometry(label_object, record["label"], reference_shape, reference_affine)
        label = np.asanyarray(label_object.dataobj).astype(np.int16, copy=False)
        regions = labelmap_to_regions(label, label_schema)

    assert reference_image is not None
    metadata = {
        "shape": list(reference_image.shape),
        "affine": reference_image.affine.tolist(),
        "header_zooms": [float(value) for value in reference_image.header.get_zooms()[:3]],
        "reference": record["images"][MODALITY_ORDER[0]],
        "label": record.get("label"),
    }
    return stacked_images, regions, metadata


def preprocess_manifest(
    raw_manifest_path: str | Path,
    output_root: str | Path,
    output_manifest_path: str | Path,
    *,
    label_schema: str,
    overwrite: bool = False,
) -> None:
    if label_schema not in LABEL_SCHEMAS:
        raise ValueError(f"unknown label schema {label_schema!r}; choose from {sorted(LABEL_SCHEMAS)}")
    raw_manifest = load_manifest(raw_manifest_path)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preprocessed_cases: list[dict[str, Any]] = []

    for record in raw_manifest["cases"]:
        case_directory = output_root / _safe_case_id(record["id"])
        image_path = case_directory / "images.npy"
        region_path = case_directory / "regions.npy"
        metadata_path = case_directory / "metadata.json"
        expected = [image_path, metadata_path]
        if record.get("label"):
            expected.append(region_path)
        if not overwrite and all(path.exists() for path in expected):
            pass
        else:
            case_directory.mkdir(parents=True, exist_ok=True)
            images, regions, metadata = load_raw_case(record, label_schema)
            np.save(image_path, images, allow_pickle=False)
            if regions is not None:
                np.save(region_path, regions, allow_pickle=False)
            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)

        processed_record: dict[str, Any] = {
            "id": record["id"],
            "image": str(image_path),
            "metadata": str(metadata_path),
            "reference": record["images"][MODALITY_ORDER[0]],
        }
        if record.get("label"):
            processed_record["regions"] = str(region_path)
            processed_record["label"] = record["label"]
        preprocessed_cases.append(processed_record)

    write_manifest(
        preprocessed_cases,
        output_manifest_path,
        metadata={
            "preprocessed": True,
            "label_schema": label_schema,
            "source_manifest": str(Path(raw_manifest_path).expanduser().resolve()),
        },
    )


def _validate_geometry(
    image: nib.spatialimages.SpatialImage,
    path: str,
    reference_shape: tuple[int, ...] | None,
    reference_affine: np.ndarray | None,
) -> None:
    if image.ndim != 3:
        raise ValueError(f"expected a 3D NIfTI image, got shape {image.shape}: {path}")
    if reference_shape is not None and image.shape != reference_shape:
        raise ValueError(f"shape mismatch for {path}: {image.shape} != {reference_shape}")
    if reference_affine is not None and not np.allclose(image.affine, reference_affine, atol=1e-4):
        raise ValueError(f"affine mismatch for {path}; modalities must be co-registered")


def _safe_case_id(case_id: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in case_id)
    if safe in {"", ".", ".."}:
        raise ValueError(f"invalid case identifier: {case_id!r}")
    return safe
