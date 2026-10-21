"""Explicit model and MRI inputs; no lookup in historical experiment directories."""

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from brats_tta.native.predictor import load_predictor

from .backends import NativeBackend, SeniorBackend
from .senior_model import UNet3D

PROFILES = ("native-in250-single-view8", "native-in250-eight-view8", "senior188-single-view8")
SENIOR_SHA = "6a563e88865e462097ad01cba14008dd3bc68c4b4b4ddafb5a3727ddb0d02d7f"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backend(profile, model, device="cuda"):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this interpreter.")
    model = Path(model).resolve()
    if profile.startswith("native-"):
        if device.type != "cuda":
            raise ValueError("Native FP16 fusion requires CUDA; CPU tests use synthetic backends.")
        predictor = load_predictor(model, device=str(device))
        predictor.network.to(device)
        backend = NativeBackend(predictor, 1 if "single" in profile else 8)
        checkpoint = model / "fold_0/checkpoint_final.pth"
    else:
        checkpoint = model
        if file_sha(checkpoint) != SENIOR_SHA:
            raise ValueError("Senior profile requires the verified best188 checkpoint.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = state["config"]
        if (
            state["epoch"] != 188
            or state["best_epoch"] != 188
            or cfg["modalities"] != list(SeniorBackend.modalities)
            or cfg["crop_size"] != [128] * 3
        ):
            raise ValueError("Senior checkpoint protocol differs from best188.")
        origin = json.loads(Path(__file__).with_name("ORIGIN.json").read_text(encoding="utf-8"))
        expected_code = next(h for p, h in origin["frozen_sources"].items() if p.endswith("/model.py"))
        if state["code_hashes"]["models/unet3d.py"] != expected_code:
            raise ValueError("Senior model code identity mismatch.")
        network = UNet3D(4, 4, tuple(cfg["features"]), dropout=cfg["dropout"])
        network.load_state_dict(state["model_state_dict"], strict=True)
        if (
            sum(p.numel() for p in network.parameters()) != 21365348
            or sum(isinstance(m, torch.nn.BatchNorm3d) for m in network.modules()) != 18
        ):
            raise ValueError("Senior model topology mismatch.")
        if not all(torch.isfinite(t).all() for t in network.state_dict().values()):
            raise ValueError("Nonfinite checkpoint state.")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("The frozen Senior188 GPU protocol requires BF16 support.")
        network.eval().requires_grad_(False).to(device)
        backend = SeniorBackend(network, torch.bfloat16 if device.type == "cuda" else torch.float32)
    identity = dict(
        profile=profile,
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha(checkpoint),
        device=str(device),
        modalities=list(backend.modalities),
        precision="native_fp16"
        if profile.startswith("native-")
        else ("bf16" if device.type == "cuda" else "fp32_cpu_validation_only"),
    )
    if profile.startswith("native-"):
        identity["metadata"] = {
            name: dict(path=str(model / name), sha256=file_sha(model / name))
            for name in ("plans.json", "dataset.json")
        }
    return backend, identity


def load_case(backend, files):
    if set(files) != {"t1c", "t1n", "t2w", "t2f"}:
        raise ValueError("Provide exactly t1c, t1n, t2w and t2f, without a label path.")
    reference = nib.load(str(files[backend.modalities[0]]))
    if len(reference.shape) != 3 or not np.allclose(reference.header.get_zooms()[:3], 1, atol=1e-6, rtol=0):
        raise ValueError("The fixed context radii require a three-dimensional 1 mm grid.")
    volumes = []
    for modality in backend.modalities:
        volume = nib.load(str(files[modality]))
        values = np.asarray(volume.dataobj, dtype=np.float32)
        if (
            values.shape != reference.shape
            or not np.isfinite(values).all()
            or not np.allclose(volume.affine, reference.affine, atol=1e-4)
        ):
            raise ValueError(f"Inconsistent MRI geometry or values: {modality}")
        volumes.append(values)
    if isinstance(backend, NativeBackend):
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

        predictor = backend.predictor
        if not np.allclose(predictor.configuration_manager.spacing, [1, 1, 1], atol=1e-6, rtol=0):
            raise ValueError("This native coordinate adapter requires unchanged 1 mm spacing.")
        raw, properties = SimpleITKIO().read_images([str(files[m]) for m in backend.modalities])
        image, _ = DefaultPreprocessor(verbose=False).run_case_npy(
            raw,
            None,
            properties,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        geometry = dict(
            shape=np.asarray(reference.shape),
            bbox=np.asarray(properties["bbox_used_for_cropping"]),
            fw=np.asarray(predictor.plans_manager.transpose_forward),
        )
        if tuple(image.shape[1:]) != tuple(np.diff(geometry["bbox"], axis=1).ravel()):
            raise ValueError("Unexpected native resampling; fixed crop restore would be invalid.")
    else:
        normalized = []
        for values in volumes:
            mask = values != 0
            if not mask.any():
                raise ValueError("All-zero modality.")
            foreground = values[mask]
            out = np.zeros_like(values)
            out[mask] = (foreground - float(foreground.mean())) / max(float(foreground.std()), 1e-6)
            normalized.append(out)
        image = np.stack(normalized)
        geometry = dict(
            shape=np.asarray(reference.shape),
            bbox=np.asarray([[0, n] for n in reference.shape]),
            fw=np.asarray([2, 1, 0]),
        )
    return image, geometry, reference
