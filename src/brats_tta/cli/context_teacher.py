"""Reconstruct class-symmetric consensus from saved native probabilities (CPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from brats_tta.native.context import teacher


def load_probability(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        value = archive[key]
    if value.ndim != 4 or value.shape[0] != 4:
        raise ValueError(f"Expected four-class (C,Z,Y,X) probabilities in {path}")
    if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"Invalid probabilities in {path}")
    if not np.allclose(value.sum(0), 1, atol=2e-5, rtol=0):
        raise ValueError(f"Probabilities do not sum to one in {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-key", default="view0", help="Array key in source NPZ (axis cache: view0)")
    parser.add_argument("--blur", required=True, type=Path, help="blur_context.npz, array key prob")
    parser.add_argument("--neutral", required=True, type=Path, help="neutral_context.npz, array key prob")
    parser.add_argument("--geometry", required=True, type=Path, help="geometry.npz containing protected")
    parser.add_argument(
        "--output", required=True, type=Path, help="New NPZ path, never overwrites an existing file"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    source = load_probability(args.source, args.source_key)
    views = [load_probability(args.blur, "prob"), load_probability(args.neutral, "prob")]
    if any(p.shape != source.shape for p in views):
        raise ValueError("All probability arrays must use the same native crop and class order.")
    with np.load(args.geometry, allow_pickle=False) as archive:
        protected = archive["protected"]
    if protected.dtype != np.bool_ or protected.shape != source.shape[1:]:
        raise ValueError("protected must be a boolean mask with the same spatial shape.")
    prediction, accepted = teacher(source.argmax(0).astype(np.uint8), views, protected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects inputs if --output aliases an input path.
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, prediction=prediction, accepted=accepted)
    print(f"Saved native-crop prediction: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
