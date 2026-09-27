from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Protocol

import nibabel as nib
import numpy as np
import torch

from brats_tta.cli.common import configure_logging, load_model_from_checkpoint
from brats_tta.data.brats import BraTSDataset
from brats_tta.metrics.segmentation import compute_region_metrics
from brats_tta.tta.inference import sliding_window_adapt_logits
from brats_tta.tta.sar import SARAdapter
from brats_tta.tta.tegda import TEGDAAdapter
from brats_tta.tta.tent import TentStepResult
from brats_tta.tta.vptta import VPTTAAdapter
from brats_tta.utils.atomic_io import atomic_write_text
from brats_tta.utils.checkpoint import load_checkpoint
from brats_tta.utils.reproducibility import resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)
METHODS = ("sar", "vptta", "tegda")
BASE_METRIC_KEYS = ("dice_ET", "dice_TC", "dice_WT", "dice_mean", "hierarchy_violation")
HD95_METRIC_KEYS = ("hd95_ET", "hd95_TC", "hd95_WT", "hd95_mean")


class AdvancedAdapter(Protocol):
    method: str
    steps: int

    def begin_case(self) -> None: ...

    def reset(self) -> None: ...

    def predict_and_adapt(self, images: torch.Tensor) -> TentStepResult: ...

    def case_diagnostics(self) -> dict[str, float | int | None]: ...

    def adaptation_state_dict(self) -> dict[str, Any]: ...

    def load_adaptation_state_dict(self, state: dict[str, Any]) -> None: ...

    def method_metadata(self) -> dict[str, Any]: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one continual 3D TTA method with resumable adaptation state."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-size", type=int, nargs=3)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--reset-per-case", action="store_true")
    parser.add_argument("--hd95", action="store_true")
    parser.add_argument("--hd95-empty-penalty", type=float, default=374.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stall-trace-seconds", type=int, default=300)

    parser.add_argument("--sar-lr", type=float, default=1e-3)
    parser.add_argument("--sar-momentum", type=float, default=0.9)
    parser.add_argument("--sar-rho", type=float, default=0.05)
    parser.add_argument("--sar-entropy-margin", type=float, default=0.4 * math.log(4.0))
    parser.add_argument("--sar-recovery-threshold", type=float, default=0.2)
    parser.add_argument("--sar-no-recovery", action="store_true")
    parser.add_argument("--sar-steps", type=int, default=1)

    parser.add_argument("--vptta-lr", type=float, default=0.05)
    parser.add_argument("--vptta-prompt-alpha", type=float, default=0.03)
    parser.add_argument("--vptta-memory-size", type=int, default=40)
    parser.add_argument("--vptta-neighbors", type=int, default=16)
    parser.add_argument("--vptta-warm-n", type=float, default=5.0)
    parser.add_argument("--vptta-steps", type=int, default=1)

    parser.add_argument("--tegda-lr", type=float, default=1e-4)
    parser.add_argument("--tegda-scope", choices=("norm", "full"), default="full")
    parser.add_argument("--tegda-dropout-samples", type=int, default=10)
    parser.add_argument("--tegda-dropout-probability", type=float, default=0.5)
    parser.add_argument("--tegda-feature-bank-size", type=int, default=10)
    parser.add_argument("--tegda-feature-percentile", type=float, default=0.90)
    parser.add_argument("--tegda-steps", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    _prepare_output_directory(output_directory, overwrite=args.overwrite)
    configure_logging(
        args.verbose,
        log_file=output_directory / "evaluation.log",
    )
    faulthandler.enable()
    seed_everything(args.seed, deterministic=args.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint has no embedded config")
    source_output_mode = checkpoint_config["model"].get("output_mode", "classes_softmax")
    source_label_schema = checkpoint_config["data"].get("label_schema", "brats_modern")
    checkpoint_metadata = {
        "completed_epoch": int(checkpoint["epoch"]) + 1,
        "source_best_dice": float(checkpoint.get("best_dice", float("nan"))),
    }
    del checkpoint

    dataset = BraTSDataset(args.manifest, training=False, output_mode="regions_sigmoid")
    case_count = min(len(dataset), args.limit) if args.limit else len(dataset)
    if case_count <= 0:
        raise ValueError("target manifest has no cases to evaluate")
    case_order = [str(dataset.cases[index]["id"]) for index in range(case_count)]
    manifest_hash = _file_hash(args.manifest)
    _validate_case_order(
        output_directory / "case_order.json",
        case_order=case_order,
        manifest_hash=manifest_hash,
        overwrite=args.overwrite,
    )

    model, config, _ = load_model_from_checkpoint(args.checkpoint, device)
    inference = config["inference"]
    patch_size = tuple(args.patch_size or inference["patch_size"])
    overlap = float(args.overlap if args.overlap is not None else inference.get("overlap", 0.5))
    threshold = float(
        args.threshold if args.threshold is not None else inference.get("threshold", 0.5)
    )
    if args.method == "vptta" and args.sw_batch_size != 1:
        raise ValueError("VPTTA requires --sw-batch-size 1")
    downsampling_factor = int(getattr(model, "required_divisibility", 1))
    deepest_patch_shape = tuple(size // downsampling_factor for size in patch_size)
    if math.prod(deepest_patch_shape) <= 1:
        raise ValueError(
            "advanced TTA requires more than one deepest-stage voxel for batch statistics; "
            f"patch {patch_size} becomes {deepest_patch_shape}"
        )
    adapter = _build_adapter(args, model=model, patch_size=patch_size)
    method_metadata = adapter.method_metadata()

    run_settings = {
        "evaluation_version": "advanced_tta_3d_v2",
        "method": args.method,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": _file_hash(args.checkpoint),
        "patch_size": list(patch_size),
        "overlap": overlap,
        "sw_batch_size": args.sw_batch_size,
        "volume_protocol": "native_grid_sliding_window",
        "adaptation_unit": "sliding_window_patch_batch",
        "protocol_comparability": (
            "matched to this project's Source/TENT evaluation; not an exact match to "
            "TEGDA's whole-volume 128-cubed BraTS protocol"
        ),
        "threshold": threshold,
        "precision": "fp32",
        "amp": False,
        "tf32": False,
        "gaussian_weighting": inference.get("gaussian_weighting", True),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "reset_per_case": args.reset_per_case,
        "case_count": case_count,
        "source_output_mode": source_output_mode,
        "source_label_schema": source_label_schema,
        "target_label_schema": dataset.label_schema,
        "target_representation": "regions_et_tc_wt",
        "hd95": args.hd95,
        "method_settings": method_metadata,
    }
    if args.hd95:
        run_settings.update(
            {
                "hd95_units": "mm",
                "hd95_definition": "area_weighted_symmetric_robust_hausdorff_95",
                "hd95_implementation": "deepmind_surface_distance_0.1",
                "hd95_empty_both": 0.0,
                "hd95_empty_one": args.hd95_empty_penalty,
                "spacing_source": "reference_nifti_header_zooms",
            }
        )
    _validate_json(
        output_directory / "run_settings.json",
        run_settings,
        overwrite=args.overwrite,
    )

    records_path = output_directory / "cases.jsonl"
    records = _load_records(records_path)
    completed_ids = [str(record["id"]) for record in records]
    if completed_ids != case_order[: len(completed_ids)]:
        raise ValueError("existing advanced-TTA records are not a prefix of the fixed case order")
    _restore_adapter_state(
        adapter,
        output_directory=output_directory,
        completed_case_ids=completed_ids,
        device=device,
    )

    def progress(stage: str, **details: Any) -> None:
        payload = {
            "pid": os.getpid(),
            "method": args.method,
            "stage": stage,
            "updated_at_unix": time.time(),
            "completed_cases": len(records),
            "expected_cases": case_count,
            **details,
        }
        _write_json(output_directory / "progress.json", payload)
        if args.stall_trace_seconds > 0:
            faulthandler.dump_traceback_later(args.stall_trace_seconds, repeat=True)

    LOGGER.info(
        "method=%s cases=%d patch=%s overlap=%.3f reset_per_case=%s device=%s resumed=%d",
        args.method,
        case_count,
        patch_size,
        overlap,
        args.reset_per_case,
        device,
        len(records),
    )
    invocation_start = time.perf_counter()
    for index in range(len(records), case_count):
        case_id = case_order[index]
        progress("loading_case", case_id=case_id, case_index=index)
        LOGGER.info("%s %d/%d %s: loading", args.method, index + 1, case_count, case_id)
        sample = dataset[index]
        if args.reset_per_case:
            adapter.reset()
        adapter.begin_case()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
        target = sample["target"].unsqueeze(0)
        if target.shape[1] == 0:
            raise ValueError(f"case {case_id} has no label")

        def patch_progress(done: int, total: int) -> None:
            progress(
                "adapting",
                case_id=case_id,
                case_index=index,
                patch_batches_done=done,
                patch_batches_total=total,
            )
            if done == 1 or done % 4 == 0 or done == total:
                LOGGER.info("%s %s: patch batches %d/%d", args.method, case_id, done, total)

        _synchronize(device)
        case_start = time.perf_counter()
        logits, adaptation = sliding_window_adapt_logits(
            model,
            adapter,
            image,
            patch_size=patch_size,
            overlap=overlap,
            sw_batch_size=args.sw_batch_size,
            gaussian_weighting=inference.get("gaussian_weighting", True),
            progress_callback=patch_progress,
        )
        _synchronize(device)
        elapsed = time.perf_counter() - case_start
        progress("metrics", case_id=case_id, case_index=index)
        metrics = compute_region_metrics(
            logits.cpu(),
            target,
            threshold=threshold,
            output_mode=source_output_mode,
            label_schema=source_label_schema,
            spacing=_nifti_spacing(sample["reference"]) if args.hd95 else None,
            include_hd95=args.hd95,
            hd95_empty_penalty=args.hd95_empty_penalty,
        )
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        record: dict[str, Any] = {
            "id": case_id,
            "index": index,
            "method": args.method,
            "seconds": elapsed,
            "peak_gpu_memory_bytes": peak_memory,
            **adaptation,
            **adapter.case_diagnostics(),
            **metrics,
        }
        next_completed_ids = [*completed_ids, case_id]
        _save_pending_adapter_state(
            adapter,
            output_directory=output_directory,
            completed_case_ids=next_completed_ids,
        )
        _append_jsonl(records_path, record)
        os.replace(
            output_directory / "adapter_state.pending.pt",
            output_directory / "adapter_state.pt",
        )
        records.append(record)
        completed_ids.append(case_id)
        progress("case_complete", case_id=case_id, case_index=index)
        LOGGER.info(
            "%s %d/%d %s: Dice=%.4f HD95=%s seconds=%.1f peak=%.2fGiB",
            args.method,
            index + 1,
            case_count,
            case_id,
            metrics["dice_mean"],
            f"{metrics['hd95_mean']:.2f}mm" if args.hd95 else "disabled",
            elapsed,
            peak_memory / (1024**3),
        )
        del image, target, logits, sample
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "method": args.method,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        **checkpoint_metadata,
        "case_count": len(records),
        "expected_case_count": case_count,
        "complete": len(records) == case_count,
        "device": str(device),
        "method_settings": method_metadata,
        "source_output_mode": source_output_mode,
        "source_label_schema": source_label_schema,
        "target_label_schema": dataset.label_schema,
        "preprocessing": dataset.manifest.get("brain_extraction"),
        "run_settings": run_settings,
        "metrics_mean": _aggregate(records, np.mean, include_hd95=args.hd95),
        "metrics_std": _aggregate(records, np.std, include_hd95=args.hd95),
        "metrics_median": _aggregate(records, np.median, include_hd95=args.hd95),
        "total_case_seconds": float(sum(float(record["seconds"]) for record in records)),
        "invocation_seconds": time.perf_counter() - invocation_start,
    }
    _write_json(output_directory / "summary.json", summary)
    progress("complete")
    if args.stall_trace_seconds > 0:
        faulthandler.cancel_dump_traceback_later()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def _build_adapter(
    args: argparse.Namespace,
    *,
    model: torch.nn.Module,
    patch_size: tuple[int, int, int],
) -> AdvancedAdapter:
    if args.method == "sar":
        return SARAdapter(
            model,
            learning_rate=args.sar_lr,
            momentum=args.sar_momentum,
            rho=args.sar_rho,
            entropy_margin=args.sar_entropy_margin,
            recovery_threshold=(None if args.sar_no_recovery else args.sar_recovery_threshold),
            steps=args.sar_steps,
        )
    if args.method == "vptta":
        return VPTTAAdapter(
            model,
            patch_size=patch_size,
            learning_rate=args.vptta_lr,
            prompt_alpha=args.vptta_prompt_alpha,
            memory_size=args.vptta_memory_size,
            neighbors=args.vptta_neighbors,
            warm_n=args.vptta_warm_n,
            steps=args.vptta_steps,
        )
    if args.method == "tegda":
        return TEGDAAdapter(
            model,
            learning_rate=args.tegda_lr,
            dropout_samples=args.tegda_dropout_samples,
            dropout_probability=args.tegda_dropout_probability,
            feature_bank_size=args.tegda_feature_bank_size,
            feature_percentile=args.tegda_feature_percentile,
            scope=args.tegda_scope,
            steps=args.tegda_steps,
        )
    raise ValueError(f"unknown method: {args.method}")


def _prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    if not overwrite:
        return
    for name in (
        "adapter_state.pt",
        "adapter_state.pending.pt",
        "case_order.json",
        "cases.jsonl",
        "evaluation.log",
        "progress.json",
        "run_settings.json",
        "summary.json",
    ):
        (path / name).unlink(missing_ok=True)


def _validate_case_order(
    path: Path,
    *,
    case_order: list[str],
    manifest_hash: str,
    overwrite: bool,
) -> None:
    payload = {"manifest_sha256": manifest_hash, "case_ids": case_order}
    _validate_json(path, payload, overwrite=overwrite)


def _validate_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.is_file() and not overwrite:
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"existing settings differ: {path}; use a new output directory")
    _write_json(path, payload)


def _save_pending_adapter_state(
    adapter: AdvancedAdapter,
    *,
    output_directory: Path,
    completed_case_ids: list[str],
) -> None:
    payload = {
        "completed_case_ids": completed_case_ids,
        "adapter_state": adapter.adaptation_state_dict(),
        "rng_state": _capture_rng_state(),
    }
    _atomic_torch_save(payload, output_directory / "adapter_state.pending.pt")


def _restore_adapter_state(
    adapter: AdvancedAdapter,
    *,
    output_directory: Path,
    completed_case_ids: list[str],
    device: torch.device,
) -> None:
    if not completed_case_ids:
        return
    candidates = (
        output_directory / "adapter_state.pt",
        output_directory / "adapter_state.pending.pt",
    )
    for path in candidates:
        if not path.is_file():
            continue
        payload = _torch_load(path)
        if payload.get("completed_case_ids") != completed_case_ids:
            continue
        adapter.load_adaptation_state_dict(payload["adapter_state"])
        _restore_rng_state(payload["rng_state"], device=device)
        if path.name.endswith("pending.pt"):
            os.replace(path, output_directory / "adapter_state.pt")
        LOGGER.info("restored adaptation state after %d cases", len(completed_case_ids))
        return
    raise ValueError("records exist but no matching resumable adapter state was found")


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any], *, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid adapter state: {path}")
    return payload


def _aggregate(
    records: list[dict[str, Any]],
    reducer: Any,
    *,
    include_hd95: bool,
) -> dict[str, float]:
    if not records:
        return {}
    keys = BASE_METRIC_KEYS + (HD95_METRIC_KEYS if include_hd95 else ())
    return {key: float(reducer([float(record[key]) for record in records])) for key in keys}


def _nifti_spacing(reference_path: str | Path) -> tuple[float, float, float]:
    zooms = nib.load(str(reference_path)).header.get_zooms()[:3]
    spacing = tuple(float(value) for value in zooms)
    if len(spacing) != 3 or not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
        raise ValueError(f"invalid NIfTI spacing {spacing}: {reference_path}")
    return spacing


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record["id"])
            if case_id in seen:
                raise ValueError(f"duplicate case in advanced-TTA records: {case_id}")
            seen.add(case_id)
            records.append(record)
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
