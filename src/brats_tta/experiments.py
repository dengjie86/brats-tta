"""Read-only experiment catalog. Paths are references, never executable imports."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from brats_tta.paths import project_root

STATUSES = {"completed", "paused", "interrupted", "planned"}
PATH_FIELDS = {"directory", "entrypoint", "protocol", "report"}
TEXT_FIELDS = {"id", "title", "model", "status", "recorded_on", "entry_kind", "note"}
FIELDS = TEXT_FIELDS | {"planned_cases", "completed_cases", "paths"}


def resolve_reference(root: Path, value: str) -> Path:
    """Resolve a portable checkout-relative reference, rejecting escapes."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Catalog paths must be nonempty strings using forward slashes.")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError(f"Catalog path must stay inside the checkout: {value}")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Catalog path escapes the checkout: {value}")
    return resolved


def load_catalog(root: str | Path | None = None) -> tuple[Path, list[dict]]:
    """Validate metadata without requiring local weights, results, or datasets."""
    base = project_root(root)
    path = base / "configs/experiments.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"schema_version", "experiments"}:
        raise ValueError("Catalog requires schema_version and experiments.")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported experiment catalog schema_version.")
    rows = data["experiments"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("experiments must be a nonempty list.")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError(f"Each experiment must have exactly these fields: {sorted(FIELDS)}")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in TEXT_FIELDS):
            raise ValueError("Experiment text fields must be nonempty strings.")
        identity = row["id"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", identity) or identity in seen:
            raise ValueError(f"Invalid or duplicate experiment id: {identity}")
        seen.add(identity)
        if row["status"] not in STATUSES:
            raise ValueError(f"Invalid status for {identity}")
        if row["entry_kind"] not in {"cache_replay", "frozen_runner"}:
            raise ValueError(f"Invalid entry_kind for {identity}")
        planned, completed = row["planned_cases"], row["completed_cases"]
        if type(planned) is not int or type(completed) is not int or not 0 <= completed <= planned:
            raise ValueError(f"Invalid case counts for {identity}")
        if planned == 0 or (row["status"] == "completed" and completed != planned):
            raise ValueError(f"Completion status and case counts disagree for {identity}")
        refs = row["paths"]
        if not isinstance(refs, dict) or set(refs) != PATH_FIELDS:
            raise ValueError(f"Invalid paths fields for {identity}")
        for value in refs.values():
            resolve_reference(base, value)
    return base, rows


def inspect_local(root: Path, row: dict) -> dict:
    """Check reference presence only, not process state or artifact integrity."""
    references = {}
    for key, relative in row["paths"].items():
        path = resolve_reference(root, relative)
        references[key] = {
            "path": str(path),
            "exists": path.is_dir() if key == "directory" else path.is_file(),
        }
    return references
