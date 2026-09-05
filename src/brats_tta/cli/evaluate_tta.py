from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brats_tta.cli.common import configure_logging, load_model_from_checkpoint
from brats_tta.data.brats import BraTSDataset
from brats_tta.engine.inference import sliding_window_logits
from brats_tta.metrics.segmentation import compute_region_metrics
from brats_tta.tta.inference import sliding_window_tent_logits
from brats_tta.tta.tent import TentAdapter, configure_norm_stats
from brats_tta.utils.reproducibility import resolve_device

LOGGER = logging.getLogger(__name__)
METHODS = ("source", "norm", "tent")
METRIC_KEYS = ("dice_ET", "dice_TC", "dice_WT", "dice_mean", "hierarchy_violation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate source, normalization-statistics, and Tent on a labeled target domain."
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
    parser.add_argument("--tent-brain-mask", action="store_true")
    parser.add_argument(
        "--continual",
        action="store_true",
        help="Carry Tent affine updates between cases; the default resets for every patient",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases (smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    device = resolve_device(args.device)
    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset = BraTSDataset(args.manifest, training=False)
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
        )


def _evaluate_method(
    method: str,
    *,
    args: argparse.Namespace,
    dataset: BraTSDataset,
    case_count: int,
    device: torch.device,
    output_directory: Path,
) -> None:
    records_path = output_directory / f"{method}_cases.jsonl"
    summary_path = output_directory / f"{method}_summary.json"
    if args.overwrite:
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    records = _load_records(records_path)
    completed_ids = {record["id"] for record in records}

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
    method_metadata: dict[str, Any] = {}
    adapter: TentAdapter | None = None
    if method == "norm":
        method_metadata.update(configure_norm_stats(model))
    elif method == "tent":
        adapter = TentAdapter(
            model,
            learning_rate=args.tent_lr,
            steps=args.tent_steps,
            use_amp=amp,
            brain_mask=args.tent_brain_mask,
        )
        method_metadata.update(
            {
                "optimizer": "Adam",
                "learning_rate": args.tent_lr,
                "steps_per_patch_batch": args.tent_steps,
                "episodic": not args.continual,
                "brain_mask": args.tent_brain_mask,
                "adapted_parameter_count": sum(p.numel() for p in adapter.parameters),
                "adapted_parameter_names": adapter.parameter_names,
            }
        )
    elif method != "source":
        raise ValueError(f"unknown method: {method}")

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
        sample = dataset[index]
        case_id = str(sample["id"])
        if case_id in completed_ids:
            continue
        if adapter is not None and not args.continual:
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
            logits, adaptation = sliding_window_tent_logits(
                model,
                adapter,
                image,
                patch_size=patch_size,
                overlap=overlap,
                sw_batch_size=sw_batch_size,
                gaussian_weighting=inference.get("gaussian_weighting", True),
            )
        _synchronize(device)
        elapsed = time.perf_counter() - case_start
        metrics = compute_region_metrics(logits.cpu(), target, threshold=threshold)
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
        del image, target, logits
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
        "metrics_mean": _aggregate(selected_records, np.mean),
        "metrics_std": _aggregate(selected_records, np.std),
        "metrics_median": _aggregate(selected_records, np.median),
        "total_case_seconds": float(sum(float(record["seconds"]) for record in selected_records)),
        "invocation_seconds": time.perf_counter() - method_start,
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    del model, adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _aggregate(records: list[dict[str, Any]], reducer: Any) -> dict[str, float]:
    if not records:
        return {}
    return {
        key: float(reducer([float(record[key]) for record in records]))
        for key in METRIC_KEYS
    }


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
