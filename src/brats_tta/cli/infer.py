from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from brats_tta.cli.common import configure_logging, load_model_from_checkpoint
from brats_tta.config import load_config
from brats_tta.data.brats import BraTSDataset
from brats_tta.engine.inference import save_brats_prediction, sliding_window_logits
from brats_tta.utils.reproducibility import resolve_device

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run source-model inference on a labeled or unlabeled manifest."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", help="Optional config override")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    device = resolve_device(args.device)
    config_override = load_config(args.config) if args.config else None
    model, config, _ = load_model_from_checkpoint(args.checkpoint, device, config_override)
    dataset = BraTSDataset(
        args.manifest,
        training=False,
        label_schema=config["data"].get("label_schema", "brats_modern"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    inference = config["inference"]
    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            case_id = batch["id"][0]
            reference = batch["reference"][0]
            logits = sliding_window_logits(
                model,
                batch["image"].to(device),
                patch_size=inference["patch_size"],
                overlap=inference.get("overlap", 0.5),
                sw_batch_size=inference.get("sw_batch_size", 1),
                gaussian_weighting=inference.get("gaussian_weighting", True),
                amp=inference.get("amp", True),
            )
            probabilities = torch.sigmoid(logits)
            save_brats_prediction(
                probabilities,
                reference,
                output_directory / f"{case_id}-seg.nii.gz",
                label_schema=config["data"].get("label_schema", "brats_modern"),
                threshold=inference.get("threshold", 0.5),
                enforce_hierarchy=inference.get("enforce_hierarchy", True),
            )
            if args.save_probabilities:
                np.savez_compressed(
                    output_directory / f"{case_id}-regions.npz",
                    probabilities=probabilities[0].cpu().numpy().astype(np.float16),
                    region_names=np.asarray(["ET", "TC", "WT"]),
                )
    LOGGER.info("Predictions written to %s", output_directory)


if __name__ == "__main__":
    main()
