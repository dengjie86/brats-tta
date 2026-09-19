"""Diagnose which regional entropy gradients control Tent adaptation.

This diagnostic never uses target labels for adaptation.  It accumulates the
gradient from every sliding-window patch and performs exactly one Adam update
per volume, avoiding the mixed-model stitching produced by patch-wise updates.
Target labels are read only after inference to report regional effects.
"""

from __future__ import annotations

import argparse
import faulthandler
import itertools
import math
import os
import time
from pathlib import Path
from typing import Callable, Sequence

import torch

from brats_tta.cli.common import configure_logging, load_model_from_checkpoint
from brats_tta.cli.evaluate_tta import _write_json
from brats_tta.cli.extract_brain_masks import _sha256, validate_destinations
from brats_tta.data.brats import BraTSDataset
from brats_tta.engine.inference import (
    _pad_for_sliding_window,
    _scan_starts,
    sliding_window_logits,
)
from brats_tta.metrics.segmentation import REGION_NAMES, compute_region_metrics
from brats_tta.tta.dense_objectives import DenseReduction, dense_entropy_loss
from brats_tta.tta.tent import TentAdapter

GradientList = list[torch.Tensor]
DIAGNOSTIC_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def _patch_batches(
    image: torch.Tensor,
    patch_size: Sequence[int],
    overlap: float,
    sw_batch_size: int,
) -> tuple[torch.Tensor, list[list[tuple[int, int, int]]]]:
    patch_size = tuple(int(value) for value in patch_size)
    padded, _ = _pad_for_sliding_window(image, patch_size)
    starts = [_scan_starts(size, patch, overlap) for size, patch in zip(padded.shape[2:], patch_size)]
    locations = list(itertools.product(*starts))
    batches = [locations[index : index + sw_batch_size] for index in range(0, len(locations), sw_batch_size)]
    return padded, batches


def accumulate_regional_volume_gradients(
    adapter: TentAdapter,
    image: torch.Tensor,
    *,
    reduction: DenseReduction,
    patch_size: Sequence[int] = (128, 128, 128),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, GradientList], dict[str, float], int]:
    """Accumulate ET, TC, WT, and exact joint gradients over one volume."""

    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError(f"expected one volume [1,C,D,H,W], got {tuple(image.shape)}")
    if not 0 <= overlap < 1 or sw_batch_size < 1:
        raise ValueError("invalid sliding-window settings")
    patch_size = tuple(int(value) for value in patch_size)
    padded, batches = _patch_batches(image, patch_size, overlap, sw_batch_size)
    if not batches:
        raise ValueError("volume produced no sliding-window batches")

    names = (*REGION_NAMES, "joint")
    gradients = {
        name: [
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for parameter in adapter.parameters
        ]
        for name in names
    }
    losses = {name: 0.0 for name in names}
    adapter.optimizer.zero_grad(set_to_none=True)

    for batch_index, locations in enumerate(batches):
        patches = torch.cat(
            [
                padded[:, :, d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]]
                for d, h, w in locations
            ],
            dim=0,
        )
        logits = adapter.model(patches)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        regional_losses = {
            region: dense_entropy_loss(
                logits[:, channel : channel + 1],
                patches,
                reduction=reduction,
                normalize=True,
            )
            for channel, region in enumerate(REGION_NAMES)
        }
        regional_losses["joint"] = dense_entropy_loss(
            logits,
            patches,
            reduction=reduction,
            normalize=True,
        )
        for loss_index, name in enumerate(names):
            loss = regional_losses[name]
            losses[name] += float(loss.detach().item()) / len(batches)
            patch_gradients = torch.autograd.grad(
                loss / len(batches),
                adapter.parameters,
                retain_graph=loss_index + 1 < len(names),
            )
            for accumulated, gradient in zip(gradients[name], patch_gradients):
                accumulated.add_(gradient.detach())
        if progress_callback is not None:
            progress_callback(batch_index + 1, len(batches))

    adapter.optimizer.zero_grad(set_to_none=True)
    return gradients, losses, len(batches)


def _flatten(gradients: GradientList) -> torch.Tensor:
    return torch.cat([gradient.reshape(-1).float() for gradient in gradients])


def regional_gradient_report(gradients: dict[str, GradientList]) -> dict:
    vectors = {name: _flatten(values) for name, values in gradients.items()}
    norms = {name: float(vector.norm().item()) for name, vector in vectors.items()}

    def cosine(left: str, right: str) -> float:
        denominator = vectors[left].norm() * vectors[right].norm()
        if denominator.item() == 0:
            return float("nan")
        return float(torch.dot(vectors[left], vectors[right]).div(denominator).item())

    pairs = (("ET", "TC"), ("ET", "WT"), ("TC", "WT"), ("joint", "WT"))
    channel_mean = (vectors["ET"] + vectors["TC"] + vectors["WT"]) / 3.0
    relative_joint_difference = float(
        ((vectors["joint"] - channel_mean).norm() / vectors["joint"].norm().clamp_min(1e-30)).item()
    )
    return {
        "l2_norm": norms,
        "cosine": {f"{left}_{right}": cosine(left, right) for left, right in pairs},
        "joint_vs_equal_channel_mean_relative_l2": relative_joint_difference,
        "parameter_elements": int(vectors["joint"].numel()),
    }


@torch.no_grad()
def apply_first_adam_step(
    adapter: TentAdapter,
    gradients: GradientList,
    *,
    learning_rate: float | None = None,
) -> float:
    """Reset to source, apply one precomputed volume gradient, and return delta L2."""

    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for group in adapter.optimizer.param_groups:
            group["lr"] = learning_rate
    adapter.reset()
    for parameter, gradient in zip(adapter.parameters, gradients):
        parameter.grad = gradient.clone()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad(set_to_none=True)
    return math.sqrt(
        sum(
            float((parameter.detach() - source).square().sum().item())
            for parameter, source in zip(adapter.parameters, adapter._source_parameters)
        )
    )


def _distribution_report(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        values,
        torch.tensor((0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0), device=values.device),
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        **{
            name: float(value.item())
            for name, value in zip(
                ("minimum", "p05", "p25", "median", "p75", "p95", "maximum"),
                quantiles,
            )
        },
    }


def prediction_report(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> dict:
    probabilities = torch.sigmoid(logits.float())
    target_bool = target >= 0.5
    prediction = probabilities >= 0.5
    if valid_mask is None:
        valid_mask = torch.ones_like(target_bool[:, :1])
    else:
        valid_mask = valid_mask.to(dtype=torch.bool, device=target_bool.device)
        if valid_mask.shape != target_bool[:, :1].shape:
            raise ValueError("valid_mask must have shape [B,1,D,H,W]")
    regions: dict[str, dict[str, float | int]] = {}
    probability_by_truth: dict[str, dict[str, dict[str, float | int]]] = {}
    for channel, name in enumerate(REGION_NAMES):
        predicted = prediction[:, channel]
        truth = target_bool[:, channel]
        intersection = int((predicted & truth).sum().item())
        predicted_count = int(predicted.sum().item())
        target_count = int(truth.sum().item())
        regions[name] = {
            "predicted_voxels": predicted_count,
            "target_voxels": target_count,
            "true_positive_voxels": intersection,
            "false_positive_voxels": predicted_count - intersection,
            "false_negative_voxels": target_count - intersection,
            "predicted_to_target_volume_ratio": (
                float(predicted_count / target_count) if target_count else float("nan")
            ),
        }
        valid = valid_mask[:, 0]
        probability_by_truth[name] = {
            "target_positive": _distribution_report(probabilities[:, channel][truth]),
            "brain_target_negative": _distribution_report(probabilities[:, channel][valid & ~truth]),
        }
    thresholds = {
        f"{threshold:.2f}": compute_region_metrics(
            probabilities,
            target,
            from_logits=False,
            threshold=threshold,
        )
        for threshold in DIAGNOSTIC_THRESHOLDS
    }
    return {
        "metrics": thresholds["0.50"],
        "regions": regions,
        "probability_by_truth": probability_by_truth,
        "threshold_sweep": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument(
        "--reduction",
        choices=("all", "brain", "foreground_background_balanced"),
        default="foreground_background_balanced",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--sweep-learning-rates",
        nargs="*",
        type=float,
        default=[],
        help="extra first-step learning rates evaluated from the same source gradients",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=(*REGION_NAMES, "joint"),
        default=(*REGION_NAMES, "joint"),
        help="objectives to evaluate after the primary and swept first steps",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging()
    faulthandler.enable()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    learning_rates = list(dict.fromkeys((args.learning_rate, *args.sweep_learning_rates)))
    if any(learning_rate <= 0 for learning_rate in learning_rates):
        raise ValueError("all learning rates must be positive")
    device = torch.device("cuda")
    dataset = BraTSDataset(args.manifest, training=False)
    lookup = {str(case["id"]): index for index, case in enumerate(dataset.cases)}
    missing = sorted(set(args.case_ids) - set(lookup))
    if missing:
        raise ValueError(f"case IDs missing from manifest: {missing}")

    root = Path(args.output_dir).resolve()
    validate_destinations(dataset.manifest, root, root / "results.json")
    root.mkdir(parents=True, exist_ok=True)
    settings = {
        "protocol": "wt_regional_gradient_volume_step_v1",
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "case_ids": args.case_ids,
        "reduction": args.reduction,
        "learning_rate": args.learning_rate,
        "learning_rate_sweep": learning_rates,
        "evaluated_objectives": args.objectives,
        "optimizer": "Adam_first_step",
        "updated_parameters": "instance_norm_affine",
        "adaptation": "all_patch_gradients_accumulated_then_one_step_per_volume",
        "labels_used_for_adaptation": False,
        "precision": "fp32",
        "tf32": False,
        "patch_size": [128, 128, 128],
        "overlap": 0.5,
        "sw_batch_size": 1,
        "thresholds_for_diagnosis": list(DIAGNOSTIC_THRESHOLDS),
    }
    _write_json(root / "run_settings.json", settings)
    records: list[dict] = []
    model, _, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    del checkpoint
    adapter = TentAdapter(
        model,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        steps=1,
        use_amp=False,
        normalize_entropy=True,
        objective_reduction=args.reduction,
    )

    def progress(stage: str, **details) -> None:
        _write_json(
            root / "progress.json",
            {
                "stage": stage,
                "pid": os.getpid(),
                "updated_at_unix": time.time(),
                "completed_cases": len(records),
                "expected_cases": len(args.case_ids),
                **details,
            },
        )

    for case_id in args.case_ids:
        adapter.reset()
        sample = dataset[lookup[case_id]]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0)
        valid_mask = image.detach().abs().sum(dim=1, keepdim=True).gt(0).cpu()
        if target.shape[1] != 3:
            raise ValueError("diagnostic requires ET/TC/WT labels for reporting only")
        started = time.perf_counter()
        progress("source_prediction", case_id=case_id)
        with torch.no_grad():
            source_logits = sliding_window_logits(
                model,
                image,
                patch_size=(128, 128, 128),
                overlap=0.5,
                sw_batch_size=1,
                amp=False,
            ).cpu()
        source_report = prediction_report(source_logits, target, valid_mask=valid_mask)
        del source_logits

        progress("regional_gradients", case_id=case_id)
        gradients, losses, patch_batches = accumulate_regional_volume_gradients(
            adapter,
            image,
            reduction=args.reduction,
            progress_callback=lambda done, total: progress(
                "regional_gradients",
                case_id=case_id,
                patch_done=done,
                patch_total=total,
            ),
        )
        sweep = {}
        for learning_rate in learning_rates:
            variants = {}
            for name in args.objectives:
                progress(
                    "post_prediction",
                    case_id=case_id,
                    objective=name,
                    learning_rate=learning_rate,
                )
                parameter_delta = apply_first_adam_step(
                    adapter,
                    gradients[name],
                    learning_rate=learning_rate,
                )
                with torch.no_grad():
                    post_logits = sliding_window_logits(
                        model,
                        image,
                        patch_size=(128, 128, 128),
                        overlap=0.5,
                        sw_batch_size=1,
                        amp=False,
                    ).cpu()
                report = prediction_report(post_logits, target, valid_mask=valid_mask)
                report["parameter_delta_l2"] = parameter_delta
                variants[name] = report
                del post_logits
            sweep[f"{learning_rate:.12g}"] = variants
        adapter.reset()
        records.append(
            {
                "id": case_id,
                "source": source_report,
                "entropy_loss": losses,
                "gradient": regional_gradient_report(gradients),
                "patch_batches": patch_batches,
                "one_step_variants": sweep[f"{args.learning_rate:.12g}"],
                "learning_rate_sweep": sweep,
                "seconds": time.perf_counter() - started,
            }
        )
        _write_json(root / "results.json", {"settings": settings, "records": records})
        del image, target, valid_mask, sample, gradients
        torch.cuda.empty_cache()
    progress("complete")


if __name__ == "__main__":
    main()
