"""Generate image-only HD-BET masks outside the read-only source dataset."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from brats_tta.data.manifest import load_manifest, write_manifest
from brats_tta.data.preprocessing import _safe_case_id, load_brain_mask


def _json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_destinations(manifest: dict, output_root: Path, output_manifest: Path) -> None:
    inputs = [Path(p).resolve() for case in manifest["cases"]
              for p in [*case["images"].values(), *([case["label"]] if case.get("label") else [])]]
    source_root = Path(manifest.get("dataset_root") or os.path.commonpath(
        [str(p.parent) for p in inputs])).resolve()
    for output in (output_root.resolve(), output_manifest.resolve()):
        if output.is_relative_to(source_root) or source_root.is_relative_to(output):
            raise ValueError("derived outputs must be outside the source dataset tree")
        if output in inputs:
            raise ValueError("output would overwrite an input")


def create_predictor(weights: Path, device: str, mirroring: bool):
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    except ImportError as exc:
        raise RuntimeError(
            "Install the brain-extraction optional dependencies in a separate environment"
        ) from exc

    class Float32Predictor(nnUNetPredictor):
        @torch.inference_mode()
        def _internal_predict_sliding_window_return_logits(self, data, slicers, do_on_device=True):
            # nnU-Net's default accumulator is FP16 even when the network uses
            # FP32. Retain its slicers, Gaussian and mirroring, but use FP32
            # for the forward AND accumulation to honor this experiment's precision.
            result_device = self.device if do_on_device else torch.device("cpu")
            accumulated = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                      dtype=torch.float32, device=result_device)
            counts = torch.zeros(data.shape[1:], dtype=torch.float32, device=result_device)
            importance = compute_gaussian(tuple(self.configuration_manager.patch_size),
                sigma_scale=1 / 8, value_scaling_factor=10, dtype=torch.float32, device=result_device)
            with torch.autocast(self.device.type, enabled=False):
                for location in slicers:
                    patch = data[location][None].to(self.device, dtype=torch.float32)
                    predicted = self._internal_maybe_mirror_and_predict(patch)[0].to(result_device)
                    if predicted.dtype != torch.float32 or not torch.isfinite(predicted).all():
                        raise RuntimeError("HD-BET prediction must be finite FP32")
                    accumulated[location] += predicted * importance
                    counts[location[1:]] += importance
            return accumulated / counts.clamp_min(torch.finfo(torch.float32).tiny)

    os.environ["nnUNet_compile"] = "F"
    predictor = Float32Predictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=mirroring,
        perform_everything_on_device=False, device=torch.device(device), verbose=False,
        verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(weights), "all")
    predictor.network.float().eval()
    return predictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--weights", required=True, help="HD-BET release_2.0.0 directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modality", choices=("t1n", "t1c", "t2w", "t2f"), default="t1c")
    parser.add_argument("--mirroring", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    source_manifest = Path(args.manifest).resolve()
    manifest = load_manifest(source_manifest)
    output_root = Path(args.output_root).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    if source_manifest == output_manifest:
        raise ValueError("the original manifest must not be overwritten")
    validate_destinations(manifest, output_root, output_manifest)
    weights = Path(args.weights).resolve()
    weight_file = weights / "fold_all/checkpoint_final.pth"
    if not weight_file.is_file():
        raise FileNotFoundError(weight_file)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    protocol = dict(method="HD-BET", hd_bet_version=importlib.metadata.version("HD-BET"),
        nnunet_version=importlib.metadata.version("nnunetv2"), weights_sha256=_sha256(weight_file),
        modality=args.modality, mirroring=args.mirroring, precision="fp32", tf32=False,
        application="same mask for all four modalities before nonzero z-score; labels unchanged")
    protocol_path = output_root / "protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("existing masks use another protocol; choose a new output root")
    _json(protocol_path, protocol)
    cases = manifest["cases"][:args.limit] if args.limit else manifest["cases"]
    predictor = None
    completed = []
    status = dict(status="running", expected_cases=len(cases), completed_cases=0)
    try:
        for index, case in enumerate(cases):
            start = time.perf_counter()
            case_id = _safe_case_id(case["id"])
            image_path = Path(case["images"][args.modality])
            mask_path = output_root / f"{case_id}_brain_mask.nii.gz"
            metadata_path = output_root / f"{case_id}.json"
            stat = image_path.stat()
            identity = dict(input=str(image_path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            exists = mask_path.is_file() and metadata_path.is_file()
            if exists:
                metadata = json.loads(metadata_path.read_text())
                if metadata["input_identity"] != identity or metadata["mask_sha256"] != _sha256(mask_path):
                    raise ValueError(f"cached mask provenance mismatch: {case_id}")
            else:
                if mask_path.exists() or metadata_path.exists():
                    raise ValueError(f"incomplete cached case requires inspection: {case_id}")
                if predictor is None:
                    predictor = create_predictor(weights, args.device, args.mirroring)
                reader = predictor.plans_manager.image_reader_writer_class()
                image, properties = reader.read_images([str(image_path)])
                # The brain extractor receives one MRI and no tumor annotation.
                mask = predictor.predict_single_npy_array(image, properties)
                reader.write_seg(mask.astype(np.uint8), str(mask_path), properties)
            reference = nib.load(str(image_path))
            mask = load_brain_mask(mask_path, reference.shape, reference.affine)
            fraction = float(mask.mean())
            if not 0.03 <= fraction <= 0.60:
                raise ValueError(f"brain-mask size requires image-only QC: {case_id}, fraction={fraction}")
            current_stat = image_path.stat()
            if (current_stat.st_size, current_stat.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError("input MRI changed during brain extraction")
            metadata = dict(input_identity=identity, mask_sha256=_sha256(mask_path),
                brain_voxels=int(mask.sum()), brain_fraction=fraction,
                brain_volume_ml=float(mask.sum()*abs(np.linalg.det(reference.affine[:3,:3]))/1000),
                geometry_verified=True, tumor_labels_used=False)
            _json(metadata_path, metadata)
            completed.append({**case, "brain_mask": str(mask_path)})
            status.update(completed_cases=len(completed), last_case=case["id"])
            _json(output_root / "status.json", status)
            print(f"{index+1}/{len(cases)} {case['id']}: brain={fraction:.3f} "
                  f"seconds={time.perf_counter()-start:.1f} reused={exists}", flush=True)
        source_metadata = {k:v for k,v in manifest.items() if k not in ("cases", "version", "modalities")}
        write_manifest(completed, output_manifest, metadata={**source_metadata,
            "brain_extraction": protocol, "source_manifest": str(source_manifest),
            "preprocessing_complete": True})
        status["status"] = "complete"
        _json(output_root / "status.json", status)
    except BaseException as exc:
        status.update(status="failed", error=str(exc))
        _json(output_root / "status.json", status)
        raise


if __name__ == "__main__":
    main()
