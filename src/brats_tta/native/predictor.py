"""Load the verified native IN250 four-class source, without experiment imports.

This loader intentionally validates its specific protocol; it is not a generic
BN or arbitrary-epoch loader. CPU loading is supported for structural checks.
"""

import hashlib
import json
from pathlib import Path

import torch


def state_sha(network):
    h = hashlib.sha256()
    for name, value in network.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_predictor(model_dir, device="cuda"):
    import os

    os.environ["nnUNet_compile"] = "false"
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / "fold_0/checkpoint_final.pth", map_location="cpu", weights_only=False)
    plans = PlansManager(str(model_dir / "plans.json"))
    dataset = json.loads((model_dir / "dataset.json").read_text(encoding="utf-8"))
    config = plans.get_configuration(checkpoint["init_args"]["configuration"])
    assert checkpoint["current_epoch"] == 250
    assert checkpoint["trainer_name"] == "nnUNetTrainerBraTS250"
    assert dataset["labels"] == {"background": 0, "NCR_NET": 1, "edema": 2, "ET": 3}
    assert list(dataset["channel_names"].values()) == ["FLAIR", "T1", "T1ce", "T2"]
    assert plans.plans["image_reader_writer"] == "SimpleITKIO"
    assert config.patch_size == [128, 128, 128]
    # The supplied trainer inherits this exact native architecture builder.
    network = get_network_from_plans(
        config.network_arch_class_name,
        config.network_arch_init_kwargs,
        config.network_arch_init_kwargs_req_import,
        4,
        4,
        allow_init=False,
        deep_supervision=False,
    )
    network.decoder.deep_supervision = False
    network.load_state_dict(checkpoint["network_weights"], strict=True)
    network.requires_grad_(False)
    network.eval()
    assert sum(p.numel() for p in network.parameters()) == 31199796
    assert not any(isinstance(m, torch.nn.BatchNorm3d) for m in network.modules())
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    # Native manual initialization avoids the old loader's torch.load default;
    # preprocessing, sliding windows, AMP and export remain upstream code.
    predictor.manual_initialization(
        network,
        plans,
        config,
        [checkpoint["network_weights"]],
        dataset,
        checkpoint["trainer_name"],
        tuple(checkpoint["inference_allowed_mirroring_axes"]),
    )
    assert predictor.allowed_mirroring_axes == (0, 1, 2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return predictor
