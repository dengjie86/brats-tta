from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from brats_tta.cli.common import configure_logging, load_model_from_checkpoint
from brats_tta.config import load_config
from brats_tta.data.brats import BraTSDataset
from brats_tta.engine.inference import sliding_window_logits
from brats_tta.metrics.segmentation import aggregate_metric_dicts, compute_region_metrics
from brats_tta.utils.reproducibility import resolve_device

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a source checkpoint on a labeled BraTS manifest.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional config override; checkpoint config is used by default")
    parser.add_argument("--manifest", help="Optional validation manifest override")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="Write aggregate metrics as JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    device = resolve_device(args.device)
    config_override = load_config(args.config) if args.config else None
    model, config, _ = load_model_from_checkpoint(args.checkpoint, device, config_override)
    manifest = args.manifest or config["data"]["val_manifest"]
    dataset = BraTSDataset(
        manifest,
        training=False,
        label_schema=config["data"].get("label_schema", "brats_modern"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    inference = config["inference"]
    results: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            target = batch["target"].to(device)
            if target.shape[1] == 0:
                raise ValueError(f"case {batch['id'][0]} has no label")
            logits = sliding_window_logits(
                model,
                batch["image"].to(device),
                patch_size=inference["patch_size"],
                overlap=inference.get("overlap", 0.5),
                sw_batch_size=inference.get("sw_batch_size", 1),
                gaussian_weighting=inference.get("gaussian_weighting", True),
                amp=inference.get("amp", True),
            )
            results.append(compute_region_metrics(logits, target, threshold=inference.get("threshold", 0.5)))
    aggregate = aggregate_metric_dicts(results)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(aggregate, file, indent=2, ensure_ascii=False)
    LOGGER.info("Evaluated %d cases", len(results))


if __name__ == "__main__":
    main()
