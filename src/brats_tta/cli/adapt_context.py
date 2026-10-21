"""Run one isolated MRI case through the maintained union2/soft_extent/no_come pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import traceback

import nibabel as nib
import numpy as np
import torch

from brats_tta.paths import project_root
from brats_tta.pipeline import run_case
from brats_tta.pipeline.io import PROFILES, file_sha, load_backend, load_case


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def runtime_identity():
    packages = {}
    for name in ("torch", "numpy", "scipy", "nibabel", "nnunetv2", "dynamic-network-architectures"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(
        python=sys.version,
        interpreter=sys.executable,
        packages=packages,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Checkout root")
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--model", required=True, help="Native model directory or Senior best188 checkpoint")
    for name in ("t1c", "t1n", "t2w", "t2f"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory under this checkout's outputs/"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    root = project_root(args.root)
    output = args.output.resolve()
    if not output.is_relative_to(root / "outputs") or output == root / "outputs":
        parser.error("--output must be a new directory below the checkout's outputs/.")
    if output.exists():
        parser.error("Output already exists; frozen or partial results will not be overwritten.")
    files = {name: getattr(args, name).resolve() for name in ("t1c", "t1n", "t2w", "t2f")}
    if not all(path.is_file() for path in files.values()):
        parser.error("All four modality files must exist.")
    if len(set(files.values())) != 4:
        parser.error("Each modality must name a distinct MRI file.")
    torch.set_num_threads(2 if args.profile.startswith("native-") else 1)
    torch.manual_seed(2025)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    try:
        backend, identity = load_backend(args.profile, args.model, args.device)
        image, geometry, reference = load_case(backend, files)
        source_root = Path(__file__).resolve().parents[1]
        source_files = [
            Path(__file__),
            *(source_root / "pipeline").glob("*.py"),
            *(source_root / "native").glob("*.py"),
        ]
        write_json(
            output / "inputs.json",
            dict(
                created_utc=datetime.now(timezone.utc).isoformat(),
                model=identity,
                runtime=runtime_identity(),
                inputs={name: dict(path=str(path), sha256=file_sha(path)) for name, path in files.items()},
                code={p.relative_to(source_root.parent).as_posix(): file_sha(p) for p in source_files},
                method="union2 + soft_extent 8 Adam steps lr=.001 rho=.5 + no_come",
                labels_used=False,
            ),
        )

        def progress(phase, step):
            print(json.dumps(dict(phase=phase, step=step), ensure_ascii=False), flush=True)

        result = run_case(backend, image, geometry, progress)
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        for name in ("prediction", "source", "teacher"):
            nib.save(nib.Nifti1Image(result[name], reference.affine, header), output / (name + ".nii.gz"))
        np.savez_compressed(output / "affine.npz", **result["affine"])
        np.savez_compressed(
            output / "teacher_additions.npz", index=result["additions"][0], probability=result["additions"][1]
        )
        write_json(
            output / "result.json",
            {key: result[key] for key in ("updates", "history", "context", "projection", "source_restored")},
        )
        write_json(
            output / "complete.json",
            dict(
                completed_utc=datetime.now(timezone.utc).isoformat(),
                labels_used=False,
                source_restored=result["source_restored"],
                files={p.name: file_sha(p) for p in output.iterdir() if p.is_file()},
            ),
        )
    except BaseException:
        write_json(
            output / "failure.json",
            dict(traceback=traceback.format_exc(), timestamp=datetime.now(timezone.utc).isoformat()),
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
