"""List/check local resources without loading a model or launching an experiment."""

from __future__ import annotations

import argparse
import json

from brats_tta.paths import resource_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Project checkout (otherwise discovered from this package)")
    parser.add_argument("--config", help="Override configs/paths.yaml")
    parser.add_argument("--check", action="store_true", help="Exit 1 if any configured resource is absent")
    args = parser.parse_args(argv)
    paths = resource_paths(args.config, args.root)
    rows = {key: {"path": str(path), "exists": path.exists()} for key, path in paths.items()}
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return int(args.check and not all(row["exists"] for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
