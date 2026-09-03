from __future__ import annotations

import argparse

import torch
from torch import nn

from brats_tta.config import load_config
from brats_tta.models import build_source_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print and optionally smoke-test the configured source model."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--forward-shape", nargs=3, type=int, metavar=("D", "H", "W"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    model = build_source_model(config["model"])
    normalization_layers = [module for module in model.modules() if isinstance(module, nn.InstanceNorm3d)]
    print(model)
    print(f"parameters: {model.parameter_count():,}")
    print(f"InstanceNorm3d layers: {len(normalization_layers)}")
    print(f"required spatial divisibility: {model.required_divisibility}")
    if args.forward_shape:
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros((1, config["model"]["in_channels"], *args.forward_shape)))
        print(f"output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
