"""Run frozen dense TENT and SAR on full PED and SSA cohorts with safe resume."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from brats_tta.cli.evaluate_tta import _validate_run_settings, _write_json
from brats_tta.cli.extract_brain_masks import validate_destinations
from brats_tta.utils.process_watchdog import run_with_progress_watchdog


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input_paths(manifests: dict[str, Path], checkpoint: Path) -> list[Path]:
    paths = [checkpoint]
    for manifest_path in manifests.values():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths.append(manifest_path)
        for case in manifest["cases"]:
            paths.extend(Path(path) for path in case["images"].values())
            paths.append(Path(case["label"]))
            if case.get("brain_mask"):
                paths.append(Path(case["brain_mask"]))
    return paths


def _snapshot(paths: list[Path]) -> dict[str, list[int]]:
    return {str(path.resolve()): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ped-manifest", "ssa-manifest", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    root = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    manifests = {
        "ped": args.ped_manifest.resolve(),
        "ssa": args.ssa_manifest.resolve(),
    }
    for manifest_path in manifests.values():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_destinations(manifest, root, root / "status.json")
    root.mkdir(parents=True, exist_ok=True)

    inputs = _input_paths(manifests, checkpoint)
    before = _snapshot(inputs)
    _validate_run_settings(root / "input_snapshot.json", before, has_records=False, overwrite=False)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )

    jobs: list[tuple[str, list[str], Path]] = []
    for domain, manifest_path in manifests.items():
        output = root / "tent" / domain
        jobs.append(
            (
                f"tent_{domain}",
                [
                    sys.executable,
                    "-m",
                    "brats_tta.cli.evaluate_tent_variant",
                    "--checkpoint",
                    str(checkpoint),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output),
                    "--objective",
                    "foreground_background_balanced",
                    "--learning-rate",
                    "0.0001",
                    "--weight-decay",
                    "0",
                    "--reset",
                    "case",
                    "--order",
                    "shuffled",
                    "--seed",
                    "1337",
                    "--corruption",
                    "none",
                ],
                output,
            )
        )
    for domain, manifest_path in manifests.items():
        output = root / "sar" / domain
        jobs.append(
            (
                f"sar_{domain}",
                [
                    sys.executable,
                    "-m",
                    "brats_tta.cli.evaluate_tegda_baseline",
                    "--checkpoint",
                    str(checkpoint),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output),
                    "--method",
                    "sar",
                    "--profile",
                    "dense_v2",
                    "--reset",
                    "case",
                    "--corruption",
                    "none",
                    "--seed",
                    "1337",
                ],
                output,
            )
        )

    status = {
        "status": "running",
        "protocol": "frozen_dense_tent_sar_full_v1",
        "started_at": _timestamp(),
        "job_order": [name for name, _, _ in jobs],
        "completed_jobs": [],
        "current_job": None,
        "pid": os.getpid(),
    }
    _write_json(root / "status.json", status)
    try:
        for name, command, output in jobs:
            status.update(current_job=name, updated_at=_timestamp())
            _write_json(root / "status.json", status)
            log_path = root / f"{name}.log"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(command, ensure_ascii=False) + "\n")
                log.flush()
                run_with_progress_watchdog(
                    command,
                    env=environment,
                    log=log,
                    progress_path=output / "progress.json",
                    timeout=300,
                    max_restarts=3,
                )
            status["completed_jobs"].append(name)
            status.update(updated_at=_timestamp())
            _write_json(root / "status.json", status)
        status.update(
            status="complete",
            current_job=None,
            finished_at=_timestamp(),
            updated_at=_timestamp(),
        )
    except BaseException as exc:
        status.update(status="failed", error=str(exc), updated_at=_timestamp())
        raise
    finally:
        unchanged = _snapshot(inputs) == before
        _write_json(
            root / "input_integrity.json",
            {
                "unchanged": unchanged,
                "files_checked": len(before),
                "check": "size and mtime_ns; inputs opened read-only",
            },
        )
        if not unchanged:
            status.update(status="failed", error="Input file identities changed")
        _write_json(root / "status.json", status)


if __name__ == "__main__":
    main()
