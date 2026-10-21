"""List and inspect recorded experiments without launching a runner."""

from __future__ import annotations

import argparse
import json

from brats_tta.experiments import inspect_local, load_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Project checkout; otherwise discovered from the package")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List recorded statuses; this is not a live process check")
    show = commands.add_parser("show", help="Show metadata and local reference availability")
    show.add_argument("id", help="Stable experiment id")
    check = commands.add_parser("check", help="Validate catalog metadata")
    check.add_argument("--local", action="store_true", help="Also require all referenced local files")
    args = parser.parse_args(argv)
    try:
        root, rows = load_catalog(args.root)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.command == "list":
        for row in rows:
            print(f"{row['id']:30} {row['status']:11} "
                  f"{row['completed_cases']}/{row['planned_cases']}  {row['model']}")
        print("Statuses are recorded snapshots; no experiment was launched or monitored.")
        return 0
    if args.command == "show":
        selected = next((row for row in rows if row["id"] == args.id), None)
        if selected is None:
            parser.error(f"Unknown experiment id: {args.id}")
        print(json.dumps({**selected, "local_references": inspect_local(root, selected)},
                         ensure_ascii=False, indent=2))
        return 0
    missing = []
    if args.local:
        for row in rows:
            missing.extend({"id": row["id"], "reference": key, **item}
                           for key, item in inspect_local(root, row).items() if not item["exists"])
    print(json.dumps({"valid": not missing, "experiments": len(rows),
                      "local_checked": args.local, "missing": missing}, ensure_ascii=False, indent=2))
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
