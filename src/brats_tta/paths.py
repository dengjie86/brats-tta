"""Project resource paths, independent of dated experiment implementations."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def project_root(root: str | Path | None = None) -> Path:
    """Find this checkout, or use an explicit root/BRATS_TTA_ROOT."""
    configured = root or os.environ.get("BRATS_TTA_ROOT")
    candidates = [Path(configured)] if configured else list(Path(__file__).resolve().parents)
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/brats_tta").is_dir():
            return candidate
    raise FileNotFoundError("Project checkout not found; pass --root or set BRATS_TTA_ROOT.")


def resource_paths(config: str | Path | None = None, root: str | Path | None = None) -> dict[str, Path]:
    """Resolve relative resource values against the checkout, never the process cwd."""
    base = project_root(root)
    config_path = Path(config).expanduser().resolve() if config else base / "configs/paths.yaml"
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not values:
        raise ValueError("Resource config must be a nonempty mapping of names to paths.")
    resolved = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise ValueError("Resource names and paths must be nonempty strings.")
        path = Path(value).expanduser()
        resolved[key] = (path if path.is_absolute() else base / path).resolve()
    return resolved
