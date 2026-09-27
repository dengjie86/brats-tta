"""Evaluate the released TEGDA plain 3D U-Net checkpoint on project manifests.

This is an explicit architecture-control experiment, not a replacement for the
project source evaluator. The released model consumes SimpleITK-style Z/Y/X
arrays, so NIfTI X/Y/Z arrays are transposed before inference and transposed
back before the common ET/TC/WT metrics are calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brats_tta.cli.common import configure_logging
from brats_tta.data.brats import BraTSDataset
from brats_tta.engine.inference import sliding_window_logits
from brats_tta.metrics.segmentation import compute_region_metrics
from brats_tta.tta.inference import sliding_window_tent_logits
from brats_tta.tta.tent import TentAdapter
from brats_tta.utils.reproducibility import resolve_device

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the released TEGDA plain 3D U-Net checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-code", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=("source", "tent"), default=["source", "tent"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--tent-lr", type=float, default=1e-3)
    parser.add_argument("--tent-steps", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    device = resolve_device(args.device)
    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    model_code = Path(args.model_code).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not model_code.is_dir():
        raise NotADirectoryError(model_code)

    dataset = BraTSDataset(manifest_path, training=False, output_mode="regions_sigmoid")
    case_count = min(len(dataset), args.limit) if args.limit else len(dataset)
    if case_count <= 0:
        raise ValueError("target manifest has no cases to evaluate")

    settings = {
        "evaluation_version": "released_tegda_unet3d_architecture_control_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_code": str(model_code),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "patch_size": list(args.patch_size),
        "overlap": args.overlap,
        "sw_batch_size": args.sw_batch_size,
        "precision": "amp" if args.amp else "fp32",
        "input_orientation": "nifti_xyz_to_model_zyx",
        "normalization": "project_zscore_nonzero",
        "target_representation": "regions_et_tc_wt",
    }
    _write_json(output_directory / "run_settings.json", settings)

    for method in dict.fromkeys(args.methods):
        _evaluate_method(
            method,
            args=args,
            dataset=dataset,
            case_count=case_count,
            device=device,
            output_directory=output_directory,
            checkpoint_path=checkpoint_path,
            model_code=model_code,
        )


def _evaluate_method(
    method: str,
    *,
    args: argparse.Namespace,
    dataset: BraTSDataset,
    case_count: int,
    device: torch.device,
    output_directory: Path,
    checkpoint_path: Path,
    model_code: Path,
) -> None:
    records_path = output_directory / f"{method}_cases.jsonl"
    summary_path = output_directory / f"{method}_summary.json"
    if args.overwrite:
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    records = _load_records(records_path)
    completed = {record["id"] for record in records}

    model, parameter_count, bn_count = _load_legacy_model(checkpoint_path, model_code, device)
    adapter: TentAdapter | None = None
    method_settings: dict[str, Any] = {
        "method": method,
        "parameter_count": parameter_count,
        "batch_norm3d_count": bn_count,
    }
    if method == "tent":
        adapter = TentAdapter(
            model,
            learning_rate=args.tent_lr,
            steps=args.tent_steps,
            use_amp=args.amp,
        )
        method_settings.update(
            {
                "optimizer": "Adam",
                "learning_rate": args.tent_lr,
                "steps_per_patch_batch": args.tent_steps,
                "episodic": True,
                "adapted_parameter_count": sum(parameter.numel() for parameter in adapter.parameters),
                "adapted_parameter_names": adapter.parameter_names,
            }
        )
    elif method != "source":
        raise ValueError(f"unknown method: {method}")

    LOGGER.info(
        "legacy model method=%s cases=%d params=%d bn=%d device=%s",
        method,
        case_count,
        parameter_count,
        bn_count,
        device,
    )
    start_time = time.perf_counter()
    for index in range(case_count):
        case_id = str(dataset.cases[index]["id"])
        if case_id in completed:
            continue
        sample = dataset[index]
        if adapter is not None:
            adapter.reset()
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0)
        # The released implementation loads SimpleITK arrays as Z/Y/X.
        model_image = image.permute(0, 1, 4, 3, 2).contiguous()
        case_start = time.perf_counter()
        if adapter is None:
            model_logits = sliding_window_logits(
                model,
                model_image,
                patch_size=args.patch_size,
                overlap=args.overlap,
                sw_batch_size=args.sw_batch_size,
                gaussian_weighting=True,
                amp=args.amp,
            )
            adaptation: dict[str, float | int] = {}
        else:
            model_logits, adaptation = sliding_window_tent_logits(
                model,
                adapter,
                model_image,
                patch_size=args.patch_size,
                overlap=args.overlap,
                sw_batch_size=args.sw_batch_size,
                gaussian_weighting=True,
            )
        logits = model_logits.permute(0, 1, 4, 3, 2).contiguous()
        metrics = compute_region_metrics(
            logits.cpu(),
            target,
            threshold=0.5,
            output_mode="classes_softmax",
            label_schema="brats_modern",
        )
        record = {
            "id": case_id,
            "index": index,
            "method": method,
            "seconds": time.perf_counter() - case_start,
            **adaptation,
            **metrics,
        }
        _append_jsonl(records_path, record)
        records.append(record)
        completed.add(case_id)
        _write_json(
            output_directory / "progress.json",
            {
                "method": method,
                "completed_cases": len(completed),
                "expected_cases": case_count,
                "case_id": case_id,
                "updated_at_unix": time.time(),
            },
        )
        LOGGER.info(
            "%s %d/%d %s: mean=%.4f ET=%.4f TC=%.4f WT=%.4f",
            method,
            index + 1,
            case_count,
            case_id,
            metrics["dice_mean"],
            metrics["dice_ET"],
            metrics["dice_TC"],
            metrics["dice_WT"],
        )
        del image, model_image, target, model_logits, logits, sample
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected = [record for record in records if int(record["index"]) < case_count]
    summary = {
        "method": method,
        "case_count": len(selected),
        "expected_case_count": case_count,
        "complete": len(selected) == case_count,
        "method_settings": method_settings,
        "metrics_mean": _aggregate(selected, np.mean),
        "metrics_std": _aggregate(selected, np.std),
        "metrics_median": _aggregate(selected, np.median),
        "total_case_seconds": float(sum(float(record["seconds"]) for record in selected)),
        "invocation_seconds": time.perf_counter() - start_time,
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    del model, adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _load_legacy_model(
    checkpoint_path: Path,
    model_code: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int, int]:
    sys.path.insert(0, str(model_code))
    module = importlib.import_module("networks.unet_3D")
    model = module.unet_3D(in_channels=4, n_classes=4, is_batchnorm=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.output_mode = "classes_softmax"
    model.to(device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    bn_count = sum(isinstance(layer, torch.nn.BatchNorm3d) for layer in model.modules())
    return model, parameter_count, bn_count


def _aggregate(records: list[dict[str, Any]], reducer: Any) -> dict[str, float]:
    keys = ("dice_ET", "dice_TC", "dice_WT", "dice_mean", "hierarchy_violation")
    return {key: float(reducer([float(record[key]) for record in records])) for key in keys}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["id"])] = record
    return list(records.values())


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
