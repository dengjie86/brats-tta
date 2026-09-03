from __future__ import annotations

import argparse
import logging

from brats_tta.cli.common import configure_logging
from brats_tta.data.preprocessing import LABEL_SCHEMAS, preprocess_manifest

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize BraTS NIfTI files and create mmap-friendly .npy cases."
    )
    parser.add_argument("--manifest", required=True, help="Raw JSON manifest")
    parser.add_argument("--output-root", required=True, help="Directory for preprocessed case arrays")
    parser.add_argument("--output-manifest", required=True, help="Resulting preprocessed JSON manifest")
    parser.add_argument("--label-schema", choices=sorted(LABEL_SCHEMAS), default="brats_modern")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    preprocess_manifest(
        args.manifest,
        args.output_root,
        args.output_manifest,
        label_schema=args.label_schema,
        overwrite=args.overwrite,
    )
    LOGGER.info("Wrote preprocessed manifest to %s", args.output_manifest)


if __name__ == "__main__":
    main()
