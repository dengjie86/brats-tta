from __future__ import annotations

import argparse
import logging

from brats_tta.cli.common import configure_logging
from brats_tta.data.manifest import discover_brats_cases, split_cases, write_manifest

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan BraTS case directories and create JSON manifests.")
    parser.add_argument("--root", required=True, help="Root directory containing BraTS case folders")
    parser.add_argument("--train-output", required=True, help="Training manifest JSON")
    parser.add_argument("--val-output", help="Validation manifest JSON; required when --val-fraction > 0")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--allow-missing-label", action="store_true", help="For unlabeled inference manifests"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    if args.val_fraction > 0 and not args.val_output:
        raise SystemExit("--val-output is required when --val-fraction > 0")
    cases = discover_brats_cases(args.root, require_label=not args.allow_missing_label)
    training_cases, validation_cases = split_cases(
        cases,
        validation_fraction=args.val_fraction,
        seed=args.seed,
    )
    metadata = {"dataset_root": args.root, "split_seed": args.seed}
    write_manifest(training_cases, args.train_output, metadata={**metadata, "split": "train"})
    if args.val_output:
        write_manifest(validation_cases, args.val_output, metadata={**metadata, "split": "validation"})
    LOGGER.info(
        "Discovered %d cases: %d train, %d validation", len(cases), len(training_cases), len(validation_cases)
    )


if __name__ == "__main__":
    main()
