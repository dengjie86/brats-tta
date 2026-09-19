"""Measure label-free TTA signal collapse on dense sigmoid predictions."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brats_tta.cli.common import load_model_from_checkpoint
from brats_tta.data.brats import BraTSDataset
from brats_tta.engine.inference import _pad_for_sliding_window, _scan_starts
from brats_tta.tta.dense_objectives import (
    dense_entropy_loss,
    dense_prediction_confidence,
    image_brain_mask,
)
from brats_tta.tta.tegda_baselines import (
    _invert_spatial_augmentation,
    _random_source_compatible_augmentation,
)
from brats_tta.tta.tent import binary_prediction_entropy, configure_tent
from brats_tta.utils.atomic_io import atomic_write_text

PATCH_SIZE = (128, 128, 128)
REGIONS = ("ET", "TC", "WT")


def _model_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    logits = model(images)
    return logits[0] if isinstance(logits, (tuple, list)) else logits


def _patches(image: torch.Tensor):
    padded, _ = _pad_for_sliding_window(image, PATCH_SIZE)
    starts = [_scan_starts(size, patch, 0.5) for size, patch in zip(padded.shape[2:], PATCH_SIZE)]
    for location in itertools.product(*starts):
        d, h, w = location
        yield padded[:, :, d : d + 128, h : h + 128, w : w + 128]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _range_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _gradient_norm(
    model: torch.nn.Module,
    parameters: list[torch.nn.Parameter],
    patch: torch.Tensor,
    reduction: str,
) -> tuple[float, float]:
    model.zero_grad(set_to_none=True)
    logits = _model_logits(model, patch)
    loss = dense_entropy_loss(logits, patch, reduction=reduction, normalize=True)
    loss.backward()
    norm = math.sqrt(sum(parameter.grad.detach().square().sum().item() for parameter in parameters))
    model.zero_grad(set_to_none=True)
    return float(loss.detach().item()), float(norm)


@torch.no_grad()
def _augmentation_disagreement(
    model: torch.nn.Module,
    patch: torch.Tensor,
    source_probabilities: torch.Tensor,
    count: int,
) -> dict[str, float]:
    predictions = []
    for _ in range(count):
        augmented, transform = _random_source_compatible_augmentation(patch)
        probability = torch.sigmoid(_model_logits(model, augmented).float())
        predictions.append(_invert_spatial_augmentation(probability, transform))
    target = torch.stack(predictions).mean(dim=0)
    difference = (source_probabilities - target).abs()
    brain = image_brain_mask(patch).expand_as(difference)
    return {
        "teacher_probability_mae_all": float(difference.mean().item()),
        "teacher_probability_mae_brain": float(
            difference[brain].mean().item() if brain.any() else difference.mean().item()
        ),
        "teacher_probability_max_change": float(difference.max().item()),
        "voxels_crossing_half_fraction": float(
            ((source_probabilities >= 0.5) != (target >= 0.5)).float().mean().item()
        ),
    }


def _case_statistics(
    model: torch.nn.Module,
    parameters: list[torch.nn.Parameter],
    image: torch.Tensor,
    *,
    augmentation_count: int,
    augmentation_patches: int,
) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "brain_fraction": [],
        "entropy_all": [],
        "entropy_brain": [],
        "entropy_fg_bg_balanced": [],
        "confidence_all": [],
        "confidence_brain": [],
        "confidence_fg_bg_balanced": [],
        "confidence_brain_uncertain_1pct": [],
        "confidence_brain_uncertain_5pct": [],
        "sar_reliable_fraction": [],
        "sar_objective": [],
    }
    for region in REGIONS:
        values[f"predicted_positive_brain_{region}"] = []
        values[f"uncertain_01_09_brain_{region}"] = []

    gradient_diagnostics: dict[str, dict[str, float]] = {}
    augmentation_diagnostics: list[dict[str, float]] = []
    patch_count = 0
    for patch_count, patch in enumerate(_patches(image), start=1):
        if patch_count == 1:
            for reduction in ("all", "brain", "foreground_background_balanced"):
                loss, norm = _gradient_norm(model, parameters, patch, reduction)
                gradient_diagnostics[reduction] = {
                    "normalized_entropy": loss,
                    "instance_norm_affine_gradient_l2": norm,
                }
        with torch.no_grad():
            logits = _model_logits(model, patch).float()
            probabilities = torch.sigmoid(logits)
            entropy = binary_prediction_entropy(logits)
            brain = image_brain_mask(patch)
            expanded_brain = brain.expand_as(probabilities)
            values["brain_fraction"].append(float(brain.float().mean().item()))
            values["entropy_all"].append(float(entropy.mean().item()))
            values["entropy_brain"].append(
                float(entropy[expanded_brain].mean().item())
                if expanded_brain.any()
                else float(entropy.mean().item())
            )
            values["entropy_fg_bg_balanced"].append(
                float(
                    dense_entropy_loss(
                        logits,
                        patch,
                        reduction="foreground_background_balanced",
                        normalize=False,
                    ).item()
                )
            )
            for name, reduction, fraction in (
                ("confidence_all", "all", 0.05),
                ("confidence_brain", "brain", 0.05),
                ("confidence_fg_bg_balanced", "foreground_background_balanced", 0.05),
                ("confidence_brain_uncertain_1pct", "brain_uncertain", 0.01),
                ("confidence_brain_uncertain_5pct", "brain_uncertain", 0.05),
            ):
                values[name].append(
                    float(
                        dense_prediction_confidence(
                            logits, patch, reduction=reduction, uncertain_fraction=fraction
                        ).item()
                    )
                )
            voxel_entropy = entropy.mean(dim=1)
            reliable = voxel_entropy < 0.4 * math.log(2.0)
            values["sar_reliable_fraction"].append(float(reliable.float().mean().item()))
            values["sar_objective"].append(float(voxel_entropy[reliable].mean().item()))
            for channel, region in enumerate(REGIONS):
                valid = brain[:, 0]
                region_probabilities = probabilities[:, channel]
                if valid.any():
                    selected = region_probabilities[valid]
                else:
                    selected = region_probabilities.flatten()
                values[f"predicted_positive_brain_{region}"].append(
                    float((selected >= 0.5).float().mean().item())
                )
                values[f"uncertain_01_09_brain_{region}"].append(
                    float(((selected > 0.1) & (selected < 0.9)).float().mean().item())
                )
            if patch_count <= augmentation_patches and augmentation_count > 0:
                augmentation_diagnostics.append(
                    _augmentation_disagreement(model, patch, probabilities, augmentation_count)
                )

    return {
        "patch_count": patch_count,
        "patch_statistics": {name: _range_summary(items) for name, items in values.items()},
        "first_patch_gradients": gradient_diagnostics,
        "forced_augmentation": {
            name: _mean([row[name] for row in augmentation_diagnostics])
            for name in augmentation_diagnostics[0]
        }
        if augmentation_diagnostics
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--augmentation-count", type=int, default=4)
    parser.add_argument("--augmentation-patches", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1 or args.augmentation_count < 0 or args.augmentation_patches < 0:
        raise ValueError("invalid diagnostic count")

    torch.set_num_threads(4)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 128-cubed diagnostic")
    device = torch.device("cuda")
    dataset = BraTSDataset(args.manifest, training=False)
    lookup = {str(case["id"]): index for index, case in enumerate(dataset.cases)}
    if args.case_ids:
        missing = sorted(set(args.case_ids) - set(lookup))
        if missing:
            raise ValueError(f"case IDs missing from manifest: {missing}")
        indices = [lookup[case_id] for case_id in args.case_ids]
    else:
        indices = list(range(min(len(dataset), args.limit)))

    model, _, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    del checkpoint
    parameters, parameter_names = configure_tent(model)
    records = []
    for index in indices:
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        print(f"diagnosing {sample['id']}", flush=True)
        records.append(
            {
                "id": sample["id"],
                **_case_statistics(
                    model,
                    parameters,
                    image,
                    augmentation_count=args.augmentation_count,
                    augmentation_patches=args.augmentation_patches,
                ),
            }
        )
        del image, sample
        torch.cuda.empty_cache()

    payload = {
        "protocol": "dense_tta_signal_diagnostic_v1",
        "precision": "fp32",
        "tf32": False,
        "labels_used": False,
        "manifest": str(Path(args.manifest).resolve()),
        "case_ids": [record["id"] for record in records],
        "patch_size": list(PATCH_SIZE),
        "overlap": 0.5,
        "augmentation_count": args.augmentation_count,
        "augmentation_patches_per_case": args.augmentation_patches,
        "adapted_parameter_names": parameter_names,
        "records": records,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
