"""Evaluate a compatible continual SAR or CoTTA baseline in strict FP32."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from brats_tta.cli.common import configure_logging, load_model_from_checkpoint
from brats_tta.cli.evaluate_tta import METRIC_KEYS, _validate_run_settings, _write_json
from brats_tta.cli.extract_brain_masks import _sha256, validate_destinations
from brats_tta.data.brats import BraTSDataset
from brats_tta.engine.inference import sliding_window_logits
from brats_tta.metrics.segmentation import compute_region_metrics
from brats_tta.tta.baseline_state import (
    load_baseline_state,
    save_baseline_state,
    source_parameters,
)
from brats_tta.tta.inference import sliding_window_tent_logits
from brats_tta.tta.input_corruption import apply_mri_mixed_corruption
from brats_tta.tta.tegda_baselines import build_tegda_baseline_adapter

LOGGER = logging.getLogger(__name__)


def summarize(records: list[dict], expected: int, method: str) -> dict:
    method_names = ("source", f"{method}_online", f"{method}_post")
    return {
        "completed_cases": len(records),
        "expected_cases": expected,
        "complete": len(records) == expected,
        "methods": {
            name: {
                "metrics_mean": {
                    key: float(np.mean([record[name][key] for record in records])) for key in METRIC_KEYS
                },
                "delta_mean_vs_source": float(
                    np.mean([record[name]["dice_mean"] - record["source"]["dice_mean"] for record in records])
                ),
            }
            for name in method_names
        }
        if records
        else {},
    }


def _settings(args, dataset: BraTSDataset, order: list[int], count: int) -> dict:
    common = {
        "protocol": (
            "tegda_other_baseline_compatible_v1"
            if args.profile == "compatible"
            else "dense_segmentation_tta_v2"
        ),
        "method": args.method,
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "case_order": [dataset.cases[index]["id"] for index in order[:count]],
        "seed": args.seed,
        "precision": "fp32",
        "tf32": False,
        "reset": "domain_start_only" if args.reset == "domain" else "each_case",
        "adaptation": "one_sweep_one_update_per_patch",
        "patch_size": [128, 128, 128],
        "overlap": 0.5,
        "sw_batch_size": 1,
        "gaussian_weighting": True,
        "threshold": 0.5,
        "source": "recomputed_frozen_source_parameters",
        "labels_used_for_adaptation": False,
        "exact_reference_reproduction": False,
        "retained_differences": [
            "our_source_checkpoint_and_instance_norm_architecture",
            "three_independent_sigmoid_regions_not_four_class_softmax",
            "source_compatible_preprocessing_and_native_resolution_sliding_windows",
            "one_optimizer_update_per_sliding_window_patch",
        ],
    }
    if args.profile == "dense_v2":
        common.update(
            profile="dense_v2",
            corruption=args.corruption,
            case_rng=("sha256_case_id_plus_seed" if args.reset == "case" else "continual_global_stream"),
            fixes=[
                "brain_supported_dense_statistics",
                "foreground_background_balanced_objective",
                "source_calibrated_case_reset",
            ],
        )
    if args.method == "sar" and args.profile == "compatible":
        common["method_settings"] = {
            "optimizer": "SAM_SGD",
            "learning_rate": 1e-3,
            "momentum": 0.9,
            "rho": 0.05,
            "entropy": "mean_region_bernoulli",
            "entropy_margin": 0.4 * np.log(2.0),
            "recovery_threshold": 0.02,
            "updated_parameters": "instance_norm_affine",
        }
    elif args.method == "sar":
        common["method_settings"] = {
            "optimizer": "SAM_SGD",
            "learning_rate": 1e-3,
            "momentum": 0.9,
            "rho": 0.05,
            "entropy": "brain_foreground_background_balanced_bernoulli",
            "entropy_margin": 0.4 * np.log(2.0),
            "recovery_threshold": None,
            "updated_parameters": "instance_norm_affine",
        }
        common["fixes"].append("disable_invalid_dense_entropy_recovery_threshold")
    elif args.profile == "compatible":
        common["method_settings"] = {
            "optimizer": "Adam",
            "learning_rate": 1e-5,
            "weight_decay": 0.9,
            "betas": [0.9, 0.999],
            "teacher_momentum": 0.99,
            "restore_probability": 0.1,
            "augmentation_threshold": 0.9,
            "augmentation_count": args.cotta_augmentations,
            "augmentation_mode": "legacy_port",
            "consistency": "bernoulli_cross_entropy_div_ln2",
            "updated_parameters": "all_student_parameters",
            "augmentation_note": "shape_preserving_3d_source_compatible_port",
        }
        common["retained_differences"].append(
            "source_compatible_3d_augmentations_replace_reference_torchio_zero_one_pipeline"
        )
    else:
        common["dense_augmentation"] = {
            "spatial": "independent_axis_flips_probability_0.5_no_rotation",
            "intensity": "per_modality_scale_0.9_1.1_shift_-0.1_0.1",
            "noise": "per_modality_probability_0.15_std_0.0_0.1",
            "exact_zero_background_preserved": True,
            "matches_source_training_transform": True,
        }
        common["method_settings"] = {
            "optimizer": "Adam",
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "teacher_momentum": 0.99,
            "restore_probability": 0.01,
            "augmentation_threshold": 0.9,
            "augmentation_count": args.cotta_augmentations,
            "augmentation_mode": "source_training",
            "confidence": "least_confident_1pct_brain_voxels",
            "consistency": "foreground_background_balanced_bernoulli_cross_entropy",
            "minimum_consistency_probability_mae": 1e-7,
            "updated_parameters": "all_student_parameters",
            "augmentation_note": "shape_preserving_3d_source_compatible_port",
        }
        common["fixes"].extend(
            [
                "remove_optimizer_weight_decay",
                "paper_restore_probability_0.01",
                "dense_uncertainty_augmentation_gate",
                "skip_numerical_zero_consistency_updates",
            ]
        )
    return common


def _case_seed(seed: int, case_id: str) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    return int(seed + int.from_bytes(digest[:4], "little")) % (2**63 - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--method", required=True, choices=("sar", "cotta"))
    parser.add_argument("--profile", choices=("compatible", "dense_v2"), default="compatible")
    parser.add_argument("--reset", choices=("domain", "case"), default="domain")
    parser.add_argument("--corruption", choices=("none", "mri_mixed"), default="none")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--cotta-augmentations", type=int, default=32)
    args = parser.parse_args()
    if args.profile == "compatible" and (args.reset != "domain" or args.corruption != "none"):
        raise ValueError("case reset and corruption are dense_v2 study options")
    configure_logging()
    faulthandler.enable()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("This study requires CUDA")
    device = torch.device("cuda")
    dataset = BraTSDataset(args.manifest, training=False)
    if args.case_ids:
        lookup = {str(case["id"]): index for index, case in enumerate(dataset.cases)}
        missing = sorted(set(args.case_ids) - set(lookup))
        if missing:
            raise ValueError(f"case IDs missing from manifest: {missing}")
        order = [lookup[case_id] for case_id in args.case_ids]
    else:
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    count = min(len(order), args.limit) if args.limit else len(order)
    if count < 1:
        raise ValueError("Empty cohort or invalid limit")
    order = order[:count]
    root = Path(args.output_dir).resolve()
    validate_destinations(dataset.manifest, root, root / "summary.json")
    root.mkdir(parents=True, exist_ok=True)
    settings = _settings(args, dataset, order, count)
    state_path = root / "continual_state.pt"
    _validate_run_settings(
        root / "run_settings.json", settings, has_records=state_path.exists(), overwrite=False
    )
    records: list[dict] = []
    committed_cases = 0

    def progress(stage: str, **details) -> None:
        _write_json(
            root / "progress.json",
            {
                "stage": stage,
                "pid": os.getpid(),
                "updated_at_unix": time.time(),
                "completed_cases": len(records),
                "committed_cases": committed_cases,
                "expected_cases": count,
                **details,
            },
        )
        faulthandler.dump_traceback_later(240, repeat=True)

    def report() -> None:
        _write_json(
            root / "summary.json",
            {**summarize(records, count, args.method), "settings": settings},
        )
        _write_json(root / "cases.json", records)

    try:
        progress("loading_model")
        model, _, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
        del checkpoint
        adapter = build_tegda_baseline_adapter(
            args.method,
            model,
            cotta_augmentations=args.cotta_augmentations,
            profile=args.profile,
        )
        if state_path.exists():
            records = load_baseline_state(state_path, adapter, settings)
            if [record["id"] for record in records] != settings["case_order"][: len(records)]:
                raise ValueError("Committed records do not match continual order")
        committed_cases = len(records)
        report()
        inference = {"patch_size": (128, 128, 128), "overlap": 0.5, "sw_batch_size": 1}
        for sequence_index in range(len(records), count):
            index = order[sequence_index]
            case_id = dataset.cases[index]["id"]
            if args.reset == "case":
                case_seed = _case_seed(args.seed, str(case_id))
                torch.manual_seed(case_seed)
                torch.cuda.manual_seed_all(case_seed)
            progress("loading_case", case_id=case_id)
            started = time.perf_counter()
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).to(device)
            if args.corruption == "mri_mixed":
                image = apply_mri_mixed_corruption(image, seed=_case_seed(args.seed, str(case_id)))
            target = sample["target"].unsqueeze(0)
            if target.shape[1] != 3:
                raise ValueError("Requires ET/TC/WT labels for evaluation only")
            torch.cuda.reset_peak_memory_stats(device)
            record = {"id": case_id, "sequence_index": sequence_index}

            def callback(phase: str):
                def notify(done: int, total: int) -> None:
                    torch.cuda.synchronize(device)
                    progress(
                        phase,
                        case_id=case_id,
                        patch_done=done,
                        patch_total=total,
                    )

                return notify

            with source_parameters(adapter) as source_model:
                logits = sliding_window_logits(
                    source_model,
                    image,
                    **inference,
                    amp=False,
                    progress_callback=callback("source_predict"),
                )
            record["source"] = compute_region_metrics(logits.cpu(), target, threshold=0.5)
            del logits
            torch.cuda.empty_cache()

            if args.reset == "case":
                adapter.reset()
            adapter.begin_case()
            logits, adaptation = sliding_window_tent_logits(
                model,
                adapter,
                image,
                **inference,
                progress_callback=callback(f"{args.method}_adapt_online"),
            )
            online_name = f"{args.method}_online"
            post_name = f"{args.method}_post"
            record[online_name] = compute_region_metrics(logits.cpu(), target, threshold=0.5)
            del logits
            if not all(torch.isfinite(parameter).all().item() for parameter in adapter.parameters):
                raise FloatingPointError("Nonfinite adapted parameters")
            logits = sliding_window_logits(
                adapter.post_model,
                image,
                **inference,
                amp=False,
                progress_callback=callback(f"{args.method}_post_predict"),
            )
            record[post_name] = compute_region_metrics(logits.cpu(), target, threshold=0.5)
            del logits
            adaptation["adaptation_objective"] = adaptation.pop("adaptation_entropy")
            adaptation.update(adapter.case_diagnostics())
            record.update(
                adaptation=adaptation,
                seconds=time.perf_counter() - started,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),
                parameter_delta_l2=sum(
                    (parameter.detach() - source).square().sum().item()
                    for parameter, source in zip(adapter.parameters, adapter._source_parameters)
                )
                ** 0.5,
            )
            progress("committing", case_id=case_id)
            save_baseline_state(state_path, adapter, settings, records + [record])
            records.append(record)
            committed_cases = len(records)
            report()
            progress("case_complete", case_id=case_id)
            LOGGER.info(
                "%d/%d %s source=%.5f online=%.5f post=%.5f seconds=%.1f",
                len(records),
                count,
                case_id,
                record["source"]["dice_mean"],
                record[online_name]["dice_mean"],
                record[post_name]["dice_mean"],
                record["seconds"],
            )
            del image, target, sample
            torch.cuda.empty_cache()
        progress("complete")
    except BaseException as exc:
        progress("failed", error=str(exc))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
