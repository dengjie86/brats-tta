from __future__ import annotations

import argparse
import logging

import torch

from brats_tta.cli.common import configure_logging
from brats_tta.config import apply_config_overrides, load_config, validate_config
from brats_tta.data import build_dataloaders
from brats_tta.engine import SourceTrainer
from brats_tta.losses.segmentation import build_loss
from brats_tta.models import build_source_model
from brats_tta.utils.distributed import initialize_distributed
from brats_tta.utils.reproducibility import seed_everything

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the fixed BraTS GLI source-domain 3D U-Net.")
    parser.add_argument("--config", required=True, help="Base YAML configuration")
    parser.add_argument("--resume", help="latest.pt or another project checkpoint")

    paths = parser.add_argument_group("paths")
    paths.add_argument("--train-manifest", help="Override data.train_manifest")
    paths.add_argument("--val-manifest", help="Override data.val_manifest")
    paths.add_argument("--output-dir", help="Override experiment.output_dir")

    training = parser.add_argument_group("common training overrides")
    training.add_argument("--batch-size", type=int, help="Batch size per process/GPU")
    training.add_argument("--epochs", type=int, help="Total number of epochs, not additional epochs")
    training.add_argument("--iterations-per-epoch", type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--num-workers", type=int)
    training.add_argument("--validate-every", type=int)
    training.add_argument("--save-every", type=int)
    training.add_argument("--seed", type=int)
    training.add_argument("--device", help="auto, cuda, cuda:0 or cpu")
    training.add_argument("--optimizer", choices=("sgd", "adamw"))
    training.add_argument(
        "--patch-size",
        nargs=3,
        type=int,
        metavar=("D", "H", "W"),
        help="Override both training and inference patch sizes",
    )
    training.add_argument(
        "--inference-patch-size",
        nargs=3,
        type=int,
        metavar=("D", "H", "W"),
        help="Override only inference.patch_size",
    )
    training.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable AMP with --amp/--no-amp",
    )
    training.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable deterministic algorithms",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override any existing config key; repeat this option as needed",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def apply_train_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply generic overrides first and named flags second, then validate."""

    updated = apply_config_overrides(config, args.config_overrides)
    named_overrides = {
        ("data", "train_manifest"): args.train_manifest,
        ("data", "val_manifest"): args.val_manifest,
        ("experiment", "output_dir"): args.output_dir,
        ("experiment", "seed"): args.seed,
        ("experiment", "deterministic"): args.deterministic,
        ("data", "num_workers"): args.num_workers,
        ("training", "batch_size"): args.batch_size,
        ("training", "epochs"): args.epochs,
        ("training", "iterations_per_epoch"): args.iterations_per_epoch,
        ("training", "learning_rate"): args.learning_rate,
        ("training", "validate_every"): args.validate_every,
        ("training", "save_every"): args.save_every,
        ("training", "device"): args.device,
        ("training", "optimizer"): args.optimizer,
        ("training", "amp"): args.amp,
    }
    for (section, key), value in named_overrides.items():
        if value is not None:
            updated[section][key] = value
    if args.patch_size is not None:
        updated["data"]["patch_size"] = list(args.patch_size)
        updated["inference"]["patch_size"] = list(args.patch_size)
    if args.inference_patch_size is not None:
        updated["inference"]["patch_size"] = list(args.inference_patch_size)
    validate_config(updated)
    return updated


def main() -> None:
    args = build_parser().parse_args()
    config = apply_train_cli_overrides(load_config(args.config), args)
    context = initialize_distributed(config["training"].get("device", "auto"))
    configure_logging(args.verbose, rank=context.rank)
    try:
        seed = int(config["experiment"].get("seed", 2025))
        seed_everything(
            seed + context.rank,
            deterministic=config["experiment"].get("deterministic", False),
        )
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        model = build_source_model(config["model"])
        parameter_count = model.parameter_count()
        number_of_outputs = (
            len(config["model"]["features"]) - 1 if config["model"].get("deep_supervision", True) else 1
        )
        loss_function = build_loss(config["loss"], number_of_outputs)
        training_loader, validation_loader = build_dataloaders(config, context)
        per_device_batch = int(config["training"]["batch_size"])
        LOGGER.info(
            "Runtime: device=%s, distributed=%s, world_size=%d",
            context.device,
            context.distributed,
            context.world_size,
        )
        LOGGER.info("Model parameters: %,d", parameter_count)
        LOGGER.info("Training manifest: %s", config["data"]["train_manifest"])
        LOGGER.info("Validation manifest: %s", config["data"]["val_manifest"])
        LOGGER.info("Output directory: %s", config["experiment"]["output_dir"])
        LOGGER.info(
            "Training setup: patch=%s, batch/GPU=%d, global_batch=%d, epochs=%d, iterations/epoch=%d, amp=%s",
            config["data"]["patch_size"],
            per_device_batch,
            per_device_batch * context.world_size,
            config["training"]["epochs"],
            config["training"]["iterations_per_epoch"],
            config["training"].get("amp", True),
        )
        LOGGER.info(
            "Training cases: %d; validation cases: %d",
            len(training_loader.dataset),
            len(validation_loader.dataset),
        )

        trainer = SourceTrainer(
            model=model,
            loss_function=loss_function,
            training_loader=training_loader,
            validation_loader=validation_loader,
            config=config,
            device=context.device,
            distributed_context=context,
        )
        if args.resume:
            trainer.resume(args.resume)
        trainer.fit()
    finally:
        context.close()


if __name__ == "__main__":
    main()
