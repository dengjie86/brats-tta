from __future__ import annotations

import argparse
import logging

from brats_tta.cli.common import configure_logging
from brats_tta.data.manifest import discover_brats_cases, split_cases, write_manifest
from brats_tta.data.preprocessing import LABEL_SCHEMAS

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan BraTS case directories and create JSON manifests.")
    parser.add_argument("--root", required=True, help="Root directory containing BraTS case folders")
    parser.add_argument("--train-output", required=True, help="Training manifest JSON")
    parser.add_argument("--val-output", help="Validation manifest JSON; required when --val-fraction > 0")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--split-unit",
        choices=("patient", "case"),
        default="patient",
        help="Keep longitudinal scans from the same BraTS patient in one split",
    )
    parser.add_argument(
        "--allow-missing-label", action="store_true", help="For unlabeled inference manifests"
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip incomplete case folders instead of changing or rejecting the source dataset",
    )
    parser.add_argument(
        "--id-prefix",
        help="Keep only cases whose IDs start with this prefix (useful for mixed flat archives)",
    )
    parser.add_argument(
        "--label-schema",
        choices=sorted(LABEL_SCHEMAS),
        help="Record the label convention in the generated manifest",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    if args.val_fraction > 0 and not args.val_output:
        raise SystemExit("--val-output is required when --val-fraction > 0")
    cases = discover_brats_cases(
        args.root,
        require_label=not args.allow_missing_label,
        skip_incomplete=args.skip_incomplete,
    )
    discovered_case_count = len(cases)
    if args.id_prefix:
        cases = [case for case in cases if str(case["id"]).startswith(args.id_prefix)]
        if not cases:
            raise SystemExit(f"--id-prefix matched no cases: {args.id_prefix!r}")
    training_cases, validation_cases = split_cases(
        cases,
        validation_fraction=args.val_fraction,
        seed=args.seed,
        group_by_patient=args.split_unit == "patient",
    )
    metadata = {
        "dataset_root": args.root,
        "split_seed": args.seed,
        "split_unit": args.split_unit,
        "discovered_case_count": discovered_case_count,
    }
    if args.id_prefix:
        metadata["id_prefix"] = args.id_prefix
    if args.label_schema:
        metadata["label_schema"] = args.label_schema
    write_manifest(training_cases, args.train_output, metadata={**metadata, "split": "train"})
    if args.val_output:
        write_manifest(validation_cases, args.val_output, metadata={**metadata, "split": "validation"})
    LOGGER.info(
        "Discovered %d cases: %d train, %d validation", len(cases), len(training_cases), len(validation_cases)
    )


if __name__ == "__main__":
    main()
