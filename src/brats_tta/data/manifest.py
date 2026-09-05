from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

MODALITY_ORDER = ("t1n", "t1c", "t2w", "t2f")

DEFAULT_SUFFIXES: dict[str, tuple[str, ...]] = {
    "t1n": ("-t1n.nii.gz", "_t1n.nii.gz", "_t1.nii.gz", "-t1n.nii", "_t1.nii"),
    "t1c": ("-t1c.nii.gz", "_t1c.nii.gz", "_t1ce.nii.gz", "-t1c.nii", "_t1ce.nii"),
    "t2w": ("-t2w.nii.gz", "_t2w.nii.gz", "_t2.nii.gz", "-t2w.nii", "_t2.nii"),
    "t2f": ("-t2f.nii.gz", "_t2f.nii.gz", "_flair.nii.gz", "-t2f.nii", "_flair.nii"),
    "label": (
        "-seg.nii.gz",
        "_seg.nii.gz",
        "_seg_new.nii.gz",
        "-seg.nii",
        "_seg.nii",
        "_seg_new.nii",
    ),
}


def discover_brats_cases(
    root: str | Path,
    *,
    require_label: bool = True,
    skip_incomplete: bool = False,
    suffixes: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Discover flat or nested BraTS cases without depending on a year prefix.

    In addition to standard case folders containing five NIfTI files directly,
    this supports Kaggle layouts where each modality is a directory and the
    actual NIfTI file is one level deeper.
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    suffixes = suffixes or DEFAULT_SUFFIXES
    keys = (*MODALITY_ORDER, "label")
    matches_by_ancestor: dict[Path, dict[str, list[Path]]] = {}
    for file_path in (path for path in root.rglob("*") if path.is_file()):
        key = _match_file_key(file_path, suffixes)
        if key is None:
            continue
        ancestor = file_path.parent
        while True:
            ancestor_matches = matches_by_ancestor.setdefault(
                ancestor,
                {candidate_key: [] for candidate_key in keys},
            )
            ancestor_matches[key].append(file_path)
            if ancestor == root:
                break
            ancestor = ancestor.parent

    complete: dict[Path, dict[str, list[Path]]] = {}
    meaningful: set[Path] = set()
    for directory, matches in matches_by_ancestor.items():
        modality_count = sum(bool(matches[modality]) for modality in MODALITY_ORDER)
        if modality_count >= 2:
            meaningful.add(directory)
        modalities_complete = all(len(matches[modality]) == 1 for modality in MODALITY_ORDER)
        label_count = len(matches["label"])
        label_is_valid = label_count == 1 if require_label else label_count <= 1
        if modalities_complete and label_is_valid:
            complete[directory] = matches

    # If a root contains only one case, both root and case directory have the
    # same five descendants. Keep only the deepest complete directory.
    complete_directories = set(complete)
    complete_ancestors = {
        ancestor
        for directory in complete_directories
        for ancestor in directory.parents
        if ancestor in complete_directories
    }
    case_directories = sorted(complete_directories - complete_ancestors)

    # Report incomplete leaf case groups, while ignoring aggregate dataset roots.
    meaningful_ancestors = {
        ancestor
        for directory in meaningful
        for ancestor in directory.parents
        if ancestor in meaningful
    }
    leaf_groups = meaningful - meaningful_ancestors
    errors = [
        _format_incomplete_case(directory, matches_by_ancestor[directory], require_label)
        for directory in sorted(leaf_groups)
        if directory not in complete
    ]

    cases: list[dict[str, Any]] = []
    for case_directory in case_directories:
        matches = complete[case_directory]
        record: dict[str, Any] = {
            "id": case_directory.name,
            "images": {
                modality: str(matches[modality][0].resolve()) for modality in MODALITY_ORDER
            },
        }
        if matches["label"]:
            record["label"] = str(matches["label"][0].resolve())
        cases.append(record)

    if errors and not skip_incomplete:
        preview = "\n".join(errors[:20])
        remainder = "" if len(errors) <= 20 else f"\n... and {len(errors) - 20} more"
        raise ValueError(f"incomplete BraTS cases were found:\n{preview}{remainder}")
    if not cases:
        raise ValueError(f"no BraTS cases found below {root}")
    identifiers = [case["id"] for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("case directory names must be unique across the dataset root")
    return cases


def split_cases(
    cases: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not cases:
        raise ValueError("cannot split an empty case list")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction > 0 and len(cases) < 2:
        raise ValueError("at least two cases are required for a non-empty train/validation split")
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    validation_count = int(round(len(shuffled) * validation_fraction))
    if validation_fraction > 0 and validation_count == 0:
        validation_count = 1
    if validation_fraction > 0:
        validation_count = min(validation_count, len(shuffled) - 1)
    validation_ids = {case["id"] for case in shuffled[:validation_count]}
    training = [case for case in cases if case["id"] not in validation_ids]
    validation = [case for case in cases if case["id"] in validation_ids]
    return training, validation


def write_manifest(
    cases: list[dict[str, Any]],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": 1,
        "modalities": list(MODALITY_ORDER),
        "cases": cases,
    }
    if metadata:
        payload.update(metadata)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("version") != 1:
        raise ValueError(f"unsupported manifest version in {manifest_path}")
    if manifest.get("modalities") != list(MODALITY_ORDER):
        raise ValueError(f"manifest modalities must be {MODALITY_ORDER}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"manifest contains no cases: {manifest_path}")
    return manifest


def _match_file_key(file_path: Path, suffixes: dict[str, tuple[str, ...]]) -> str | None:
    filename = file_path.name.lower()
    matching_keys = [
        key
        for key, candidate_suffixes in suffixes.items()
        if filename.endswith(tuple(suffix.lower() for suffix in candidate_suffixes))
    ]
    if len(matching_keys) > 1:
        raise ValueError(f"ambiguous BraTS filename {file_path}: matches {matching_keys}")
    return matching_keys[0] if matching_keys else None


def _format_incomplete_case(
    directory: Path,
    matches: dict[str, list[Path]],
    require_label: bool,
) -> str:
    problems: list[str] = []
    missing = [modality for modality in MODALITY_ORDER if not matches[modality]]
    if missing:
        problems.append(f"missing modalities {missing}")
    duplicates = [key for key, paths in matches.items() if len(paths) > 1]
    if duplicates:
        problems.append(f"multiple matches for {duplicates}")
    if require_label and not matches["label"]:
        problems.append("missing segmentation label")
    return f"{directory}: {', '.join(problems)}"
