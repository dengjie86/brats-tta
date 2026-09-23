from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from brats_tta.cli.common import configure_logging, load_model_from_checkpoint
from brats_tta.data.brats import BraTSDataset
from brats_tta.engine.inference import sliding_window_logits
from brats_tta.metrics.segmentation import compute_region_metrics
from brats_tta.tta.inference import sliding_window_tent_logits
from brats_tta.tta.tent import TentAdapter
from brats_tta.utils.atomic_io import atomic_write_text
from brats_tta.utils.checkpoint import load_checkpoint
from brats_tta.utils.reproducibility import resolve_device

LOGGER = logging.getLogger(__name__)
METHODS = ("source", "tent")
BASE_METRIC_KEYS = ("dice_ET", "dice_TC", "dice_WT", "dice_mean", "hierarchy_violation")
HD95_METRIC_KEYS = ("hd95_ET", "hd95_TC", "hd95_WT", "hd95_mean")
LESION_WISE_DICE_KEYS = (
    "lesionwise_dice_ET",
    "lesionwise_dice_TC",
    "lesionwise_dice_WT",
    "lesionwise_dice_mean",
)
LESION_WISE_COUNT_KEYS = tuple(
    f"lesionwise_{count}_{region}"
    for count in ("tp", "fp", "fn")
    for region in ("ET", "TC", "WT")
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the source model and episodic TENT on a labeled target domain."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-size", type=int, nargs=3)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--sw-batch-size", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tent-lr", type=float, default=1e-3)
    parser.add_argument("--tent-steps", type=int, default=1)
    parser.add_argument(
        "--tent-bn-scope",
        choices=("all", "encoder", "decoder", "shallow", "deep"),
        default="all",
        help="Which BatchNorm affine layers receive TENT updates",
    )
    parser.add_argument(
        "--tent-update-affine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update selected BatchNorm affine scale/shift parameters",
    )
    parser.add_argument(
        "--tent-use-batch-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use target-patch BatchNorm statistics instead of source running statistics",
    )
    parser.add_argument(
        "--hd95",
        action="store_true",
        help="Also compute physical-space HD95 for ET, TC and WT",
    )
    parser.add_argument(
        "--hd95-empty-penalty",
        type=float,
        default=374.0,
        help="HD95 in mm when exactly one of prediction/target is empty",
    )
    parser.add_argument(
        "--lesion-wise",
        action="store_true",
        help="Also compute official BraTS 2023 lesion-wise Dice for ET, TC and WT",
    )
    parser.add_argument("--lesion-dilation-factor", type=int, default=3)
    parser.add_argument("--lesion-volume-threshold-mm3", type=float, default=50.0)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases (smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stall-trace-seconds", type=int, default=300,
                        help="Dump the Python stack after this long without progress (0 disables)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    device = resolve_device(args.device)
    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    faulthandler.enable()
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint has no embedded config")
    source_output_mode = checkpoint_config["model"].get("output_mode", "classes_softmax")
    source_label_schema = checkpoint_config["data"].get("label_schema", "brats_modern")
    del checkpoint
    # Always load target labels as common ET/TC/WT regions.  This supports PED's
    # additional calcification class without pretending it is a source GLI class.
    dataset = BraTSDataset(args.manifest, training=False, output_mode="regions_sigmoid")
    case_count = min(len(dataset), args.limit) if args.limit else len(dataset)
    if case_count <= 0:
        raise ValueError("target manifest has no cases to evaluate")

    for method in dict.fromkeys(args.methods):
        _evaluate_method(
            method,
            args=args,
            dataset=dataset,
            case_count=case_count,
            device=device,
            output_directory=output_directory,
            source_output_mode=source_output_mode,
            source_label_schema=source_label_schema,
        )


def _evaluate_method(
    method: str,
    *,
    args: argparse.Namespace,
    dataset: BraTSDataset,
    case_count: int,
    device: torch.device,
    output_directory: Path,
    source_output_mode: str,
    source_label_schema: str,
) -> None:
    records_path = output_directory / f"{method}_cases.jsonl"
    summary_path = output_directory / f"{method}_summary.json"
    if args.overwrite:
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    records = _load_records(records_path)
    completed_ids = {record["id"] for record in records}

    def progress(stage: str, **details: Any) -> None:
        _write_json(output_directory / "progress.json", {
            "pid": os.getpid(), "method": method, "stage": stage,
            "updated_at_unix": time.time(), "completed_cases": len(completed_ids),
            "expected_cases": case_count, **details,
        })
        if args.stall_trace_seconds > 0:
            faulthandler.dump_traceback_later(args.stall_trace_seconds, repeat=True)

    progress("loading_model")

    model, config, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    checkpoint_metadata = {
        "completed_epoch": int(checkpoint["epoch"]) + 1,
        "source_best_dice": float(checkpoint.get("best_dice", float("nan"))),
    }
    del checkpoint
    inference = config["inference"]
    patch_size = tuple(args.patch_size or inference["patch_size"])
    overlap = float(args.overlap if args.overlap is not None else inference.get("overlap", 0.5))
    sw_batch_size = int(
        args.sw_batch_size if args.sw_batch_size is not None else inference.get("sw_batch_size", 1)
    )
    threshold = float(
        args.threshold if args.threshold is not None else inference.get("threshold", 0.5)
    )
    amp = bool(args.amp if args.amp is not None else inference.get("amp", True))
    if not amp:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    method_metadata: dict[str, Any] = {}
    adapter: TentAdapter | None = None
    if method == "tent":
        batch_norm_count = sum(isinstance(layer, torch.nn.BatchNorm3d) for layer in model.modules())
        allow_noop = batch_norm_count == 0
        adapter = TentAdapter(
            model,
            learning_rate=args.tent_lr,
            steps=args.tent_steps,
            use_amp=amp,
            bn_scope=args.tent_bn_scope,
            update_affine=args.tent_update_affine,
            use_batch_stats=args.tent_use_batch_stats,
            allow_noop=allow_noop,
        )
        method_metadata.update(
            {
                "optimizer": "Adam",
                "implementation": "batchnorm_affine_categorical_entropy_v1",
                "gradient_scaling": adapter.scaler.is_enabled(),
                "output_mode": adapter.output_mode,
                "learning_rate": args.tent_lr,
                "steps_per_patch_batch": args.tent_steps,
                "weight_decay": 0.0,
                "episodic": True,
                "bn_scope": args.tent_bn_scope,
                "update_affine": args.tent_update_affine,
                "use_batch_stats": args.tent_use_batch_stats,
                "adapted_parameter_count": sum(p.numel() for p in adapter.parameters),
                "adapted_parameter_names": adapter.parameter_names,
                "no_op": adapter.is_noop,
                "no_op_reason": (
                    "model has no affine BatchNorm3d layers" if adapter.is_noop else None
                ),
            }
        )
    elif method != "source":
        raise ValueError(f"unknown method: {method}")

    def file_hash(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    run_settings = {
        "evaluation_version": (
            "source_tent_4class_bn_lesionwise_brats2023_v1"
            if args.lesion_wise
            else (
                "source_tent_4class_bn_hd95_official_v2"
                if args.hd95
                else "source_tent_4class_bn_v1"
            )
        ),
        "method": method,
        "manifest_sha256": file_hash(args.manifest),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "patch_size": list(patch_size), "overlap": overlap, "sw_batch_size": sw_batch_size,
        "threshold": threshold, "amp": amp,
        "gaussian_weighting": inference.get("gaussian_weighting", True),
        "source_output_mode": source_output_mode,
        "source_label_schema": source_label_schema,
        "target_label_schema": dataset.label_schema,
        "target_representation": "regions_et_tc_wt",
        "method_settings": method_metadata,
    }
    if args.hd95:
        run_settings.update(
            {
                "hd95": True,
                "hd95_units": "mm",
                "hd95_definition": "area_weighted_symmetric_robust_hausdorff_95",
                "hd95_implementation": "deepmind_surface_distance_0.1",
                "hd95_empty_both": 0.0,
                "hd95_empty_one": float(args.hd95_empty_penalty),
                "spacing_source": "reference_nifti_header_zooms",
            }
        )
    if args.lesion_wise:
        run_settings.update(
            {
                "lesion_wise": True,
                "lesion_wise_definition": "BraTS-2023-Metrics",
                "lesion_connectivity": 26,
                "lesion_dilation_structure_connectivity": 18,
                "lesion_dilation_factor": int(args.lesion_dilation_factor),
                "lesion_volume_threshold_mm3": float(args.lesion_volume_threshold_mm3),
                "lesion_false_positive_dice": 0.0,
                "spacing_source": "reference_nifti_header_zooms",
            }
        )
    _validate_run_settings(output_directory / f"{method}_run_settings.json", run_settings,
                           has_records=bool(records), overwrite=args.overwrite)

    LOGGER.info(
        "Method=%s cases=%d patch=%s overlap=%.3f amp=%s device=%s already_complete=%d",
        method,
        case_count,
        patch_size,
        overlap,
        amp,
        device,
        len(completed_ids),
    )
    method_start = time.perf_counter()
    for index in range(case_count):
        case_id = str(dataset.cases[index]["id"])
        if case_id in completed_ids:
            continue
        progress("loading_case", case_id=case_id, case_index=index)
        LOGGER.info("%s %d/%d %s: loading", method, index + 1, case_count, case_id)
        sample = dataset[index]
        progress("preparing_inference", case_id=case_id, case_index=index)
        if adapter is not None:
            adapter.reset()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
        target = sample["target"].unsqueeze(0)
        if target.shape[1] == 0:
            raise ValueError(f"case {case_id} has no label")
        _synchronize(device)
        case_start = time.perf_counter()
        adaptation: dict[str, float | int] = {}
        progress("inference", case_id=case_id, case_index=index)
        if adapter is None:
            logits = sliding_window_logits(
                model,
                image,
                patch_size=patch_size,
                overlap=overlap,
                sw_batch_size=sw_batch_size,
                gaussian_weighting=inference.get("gaussian_weighting", True),
                amp=amp,
            )
        else:
            def patch_progress(done: int, total: int) -> None:
                progress("adapting", case_id=case_id, case_index=index,
                         patch_batches_done=done, patch_batches_total=total)
                if done == 1 or done % 6 == 0 or done == total:
                    LOGGER.info("tent %s: patch batches %d/%d", case_id, done, total)

            logits, adaptation = sliding_window_tent_logits(
                model,
                adapter,
                image,
                patch_size=patch_size,
                overlap=overlap,
                sw_batch_size=sw_batch_size,
                gaussian_weighting=inference.get("gaussian_weighting", True),
                progress_callback=patch_progress,
            )
        _synchronize(device)
        elapsed = time.perf_counter() - case_start
        progress("metrics", case_id=case_id, case_index=index)
        spacing = (
            _nifti_spacing(sample["reference"])
            if args.hd95 or args.lesion_wise
            else None
        )
        metrics = compute_region_metrics(
            logits.cpu(),
            target,
            threshold=threshold,
            output_mode=source_output_mode,
            label_schema=source_label_schema,
            spacing=spacing,
            include_hd95=args.hd95,
            hd95_empty_penalty=args.hd95_empty_penalty,
            include_lesion_wise=args.lesion_wise,
            lesion_dilation_factor=args.lesion_dilation_factor,
            lesion_volume_threshold_mm3=args.lesion_volume_threshold_mm3,
        )
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        record: dict[str, Any] = {
            "id": case_id,
            "index": index,
            "method": method,
            "seconds": elapsed,
            "peak_gpu_memory_bytes": peak_memory,
            **adaptation,
            **metrics,
        }
        _append_jsonl(records_path, record)
        records.append(record)
        completed_ids.add(case_id)
        progress("case_complete", case_id=case_id, case_index=index)
        LOGGER.info(
            "%s %d/%d %s: mean=%.4f ET=%.4f TC=%.4f WT=%.4f seconds=%.1f peak=%.2fGiB",
            method,
            index + 1,
            case_count,
            case_id,
            metrics["dice_mean"],
            metrics["dice_ET"],
            metrics["dice_TC"],
            metrics["dice_WT"],
            elapsed,
            peak_memory / (1024**3),
        )
        if args.hd95:
            LOGGER.info(
                "%s %d/%d %s: HD95 mean=%.2fmm ET=%.2f TC=%.2f WT=%.2f",
                method,
                index + 1,
                case_count,
                case_id,
                metrics["hd95_mean"],
                metrics["hd95_ET"],
                metrics["hd95_TC"],
                metrics["hd95_WT"],
            )
        if args.lesion_wise:
            LOGGER.info(
                "%s %d/%d %s: lesion-wise Dice mean=%.4f ET=%.4f TC=%.4f WT=%.4f",
                method,
                index + 1,
                case_count,
                case_id,
                metrics["lesionwise_dice_mean"],
                metrics["lesionwise_dice_ET"],
                metrics["lesionwise_dice_TC"],
                metrics["lesionwise_dice_WT"],
            )
        del image, target, logits, sample
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_ids = {str(dataset.cases[index]["id"]) for index in range(case_count)}
    selected_records = [record for record in records if record["id"] in selected_ids]
    summary = {
        "method": method,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        **checkpoint_metadata,
        "case_count": len(selected_records),
        "expected_case_count": case_count,
        "complete": len(selected_records) == case_count,
        "patch_size": list(patch_size),
        "overlap": overlap,
        "sw_batch_size": sw_batch_size,
        "threshold": threshold,
        "amp": amp,
        "device": str(device),
        "method_settings": method_metadata,
        "source_output_mode": source_output_mode,
        "source_label_schema": source_label_schema,
        "target_label_schema": dataset.label_schema,
        "preprocessing": dataset.manifest.get("brain_extraction"),
        "run_settings": run_settings,
        "metrics_mean": _aggregate(
            selected_records,
            np.mean,
            include_hd95=args.hd95,
            include_lesion_wise=args.lesion_wise,
        ),
        "metrics_std": _aggregate(
            selected_records,
            np.std,
            include_hd95=args.hd95,
            include_lesion_wise=args.lesion_wise,
        ),
        "metrics_median": _aggregate(
            selected_records,
            np.median,
            include_hd95=args.hd95,
            include_lesion_wise=args.lesion_wise,
        ),
        "total_case_seconds": float(sum(float(record["seconds"]) for record in selected_records)),
        "invocation_seconds": time.perf_counter() - method_start,
    }
    _write_json(summary_path, summary)
    progress("complete")
    if args.stall_trace_seconds > 0:
        faulthandler.cancel_dump_traceback_later()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    del model, adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _validate_run_settings(path: Path, expected: dict, *, has_records: bool, overwrite: bool) -> None:
    """Prevent silent reuse of scores from another mask, checkpoint or step count."""
    if not overwrite:
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise ValueError("Evaluation settings changed; use a new output directory")
        elif has_records:
            raise ValueError("Existing results lack provenance; use a new output directory")
    _write_json(path, expected)


def _aggregate(
    records: list[dict[str, Any]],
    reducer: Any,
    *,
    include_hd95: bool = False,
    include_lesion_wise: bool = False,
) -> dict[str, float]:
    if not records:
        return {}
    metric_keys = BASE_METRIC_KEYS + (HD95_METRIC_KEYS if include_hd95 else ())
    if include_lesion_wise:
        metric_keys += LESION_WISE_DICE_KEYS + LESION_WISE_COUNT_KEYS
    return {
        key: float(reducer([float(record[key]) for record in records]))
        for key in metric_keys
    }


def _nifti_spacing(reference_path: str | Path) -> tuple[float, float, float]:
    zooms = nib.load(str(reference_path)).header.get_zooms()[:3]
    spacing = tuple(float(value) for value in zooms)
    if len(spacing) != 3 or not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
        raise ValueError(f"invalid NIfTI spacing {spacing}: {reference_path}")
    return spacing


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                records[str(record["id"])] = record
    return list(records.values())


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
