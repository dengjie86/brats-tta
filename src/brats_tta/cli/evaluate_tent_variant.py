"""Evaluate architecture-compatible Tent objectives in strict FP32."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import logging
import math
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
from brats_tta.tta.continual_state import load_state, save_state, source_affine_parameters
from brats_tta.tta.inference import sliding_window_tent_logits
from brats_tta.tta.input_corruption import apply_mri_mixed_corruption
from brats_tta.tta.tent import TentAdapter

LOGGER = logging.getLogger(__name__)


def _summary(records: list[dict], expected: int) -> dict:
    methods = ("source", "tent_online", "tent_post")
    return {
        "completed_cases": len(records),
        "expected_cases": expected,
        "complete": len(records) == expected,
        "methods": {
            method: {
                "metrics_mean": {
                    key: float(np.mean([record[method][key] for record in records])) for key in METRIC_KEYS
                },
                "delta_mean_vs_source": float(
                    np.mean(
                        [record[method]["dice_mean"] - record["source"]["dice_mean"] for record in records]
                    )
                ),
            }
            for method in methods
        }
        if records
        else {},
    }


def _case_seed(seed: int, case_id: str) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    return int(seed + int.from_bytes(digest[:4], "little")) % (2**63 - 1)


def _ordered_indices(args, dataset: BraTSDataset) -> list[int]:
    lookup = {str(case["id"]): index for index, case in enumerate(dataset.cases)}
    if args.case_ids:
        missing = sorted(set(args.case_ids) - set(lookup))
        if missing:
            raise ValueError(f"case IDs missing from manifest: {missing}")
        indices = [lookup[case_id] for case_id in args.case_ids]
    elif args.order == "shuffled":
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    else:
        indices = list(range(len(dataset)))
    return indices[: args.limit] if args.limit else indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--objective",
        choices=("all", "brain", "foreground_background_balanced"),
        required=True,
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--reset", choices=("case", "domain"), default="domain")
    parser.add_argument("--order", choices=("manifest", "shuffled"), default="manifest")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--corruption", choices=("none", "mri_mixed"), default="none")
    args = parser.parse_args()
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer settings")

    configure_logging()
    faulthandler.enable()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dataset = BraTSDataset(args.manifest, training=False)
    order = _ordered_indices(args, dataset)
    if not order:
        raise ValueError("empty evaluation cohort")

    root = Path(args.output_dir).resolve()
    validate_destinations(dataset.manifest, root, root / "summary.json")
    root.mkdir(parents=True, exist_ok=True)
    settings = {
        "protocol": "architecture_compatible_tent_objective_v1",
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "case_order": [dataset.cases[index]["id"] for index in order],
        "seed": args.seed,
        "precision": "fp32",
        "tf32": False,
        "objective": args.objective,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": [0.9, 0.999],
        "reset": args.reset,
        "corruption": args.corruption,
        "patch_size": [128, 128, 128],
        "overlap": 0.5,
        "sw_batch_size": 1,
        "gaussian_weighting": True,
        "threshold": 0.5,
        "labels_used_for_adaptation": False,
    }
    state_path = root / "continual_state.pt"
    _validate_run_settings(
        root / "run_settings.json",
        settings,
        has_records=state_path.exists(),
        overwrite=False,
    )
    records: list[dict] = []

    def progress(stage: str, **details) -> None:
        _write_json(
            root / "progress.json",
            {
                "stage": stage,
                "pid": os.getpid(),
                "updated_at_unix": time.time(),
                "completed_cases": len(records),
                "expected_cases": len(order),
                **details,
            },
        )
        faulthandler.dump_traceback_later(240, repeat=True)

    def report() -> None:
        _write_json(
            root / "summary.json",
            {**_summary(records, len(order)), "settings": settings},
        )
        _write_json(root / "cases.json", records)

    try:
        progress("loading_model")
        model, _, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
        del checkpoint
        adapter = TentAdapter(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            steps=1,
            use_amp=False,
            normalize_entropy=True,
            objective_reduction=args.objective,
        )
        if state_path.exists():
            records = load_state(state_path, adapter, settings)
            if [record["id"] for record in records] != settings["case_order"][: len(records)]:
                raise ValueError("committed records do not match case order")
        report()
        inference = {
            "patch_size": (128, 128, 128),
            "overlap": 0.5,
            "sw_batch_size": 1,
        }
        for sequence_index in range(len(records), len(order)):
            if args.reset == "case":
                adapter.reset()
            index = order[sequence_index]
            case_id = str(dataset.cases[index]["id"])
            progress("loading_case", case_id=case_id)
            started = time.perf_counter()
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).to(device)
            if args.corruption == "mri_mixed":
                image = apply_mri_mixed_corruption(image, seed=_case_seed(args.seed, case_id))
            target = sample["target"].unsqueeze(0)
            if target.shape[1] != 3:
                raise ValueError("requires ET/TC/WT labels for evaluation only")
            torch.cuda.reset_peak_memory_stats(device)
            record = {"id": case_id, "sequence_index": sequence_index}

            def callback(phase: str):
                def notify(done: int, total: int) -> None:
                    torch.cuda.synchronize(device)
                    if done == 1 or done % 6 == 0 or done == total:
                        progress(
                            phase,
                            case_id=case_id,
                            patch_done=done,
                            patch_total=total,
                        )

                return notify

            with source_affine_parameters(adapter) as source_model:
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

            logits, adaptation = sliding_window_tent_logits(
                model,
                adapter,
                image,
                **inference,
                progress_callback=callback("tent_adapt_online"),
            )
            record["tent_online"] = compute_region_metrics(logits.cpu(), target, threshold=0.5)
            del logits
            logits = sliding_window_logits(
                model,
                image,
                **inference,
                amp=False,
                progress_callback=callback("tent_post_predict"),
            )
            record["tent_post"] = compute_region_metrics(logits.cpu(), target, threshold=0.5)
            del logits
            if not all(torch.isfinite(parameter).all().item() for parameter in adapter.parameters):
                raise FloatingPointError("nonfinite adapted affine parameters")
            record.update(
                adaptation=adaptation,
                seconds=time.perf_counter() - started,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),
                affine_delta_l2=math.sqrt(
                    sum(
                        (parameter.detach() - source).square().sum().item()
                        for parameter, source in zip(adapter.parameters, adapter._source_parameters)
                    )
                ),
            )
            progress("committing", case_id=case_id)
            save_state(state_path, adapter, settings, records + [record])
            records.append(record)
            report()
            progress("case_complete", case_id=case_id)
            LOGGER.info(
                "%d/%d %s source=%.5f online=%.5f post=%.5f seconds=%.1f",
                len(records),
                len(order),
                case_id,
                record["source"]["dice_mean"],
                record["tent_online"]["dice_mean"],
                record["tent_post"]["dice_mean"],
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
