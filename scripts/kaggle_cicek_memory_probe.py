from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KERNEL_ROOT = Path(__file__).resolve().parent
RESULT_PATH = Path("/kaggle/working/cicek_memory_probe_results.json")
SOURCE_CONFIG_NAME = "source_brats_gli_cicek_unet3d_bn.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


def locate_source() -> tuple[Path, Path]:
    """Find the current source package and its Cicek configuration."""

    roots = [KERNEL_ROOT, Path("/kaggle/working"), Path("/kaggle/input")]
    for root in roots:
        if not root.exists():
            continue
        package_inits = sorted(root.rglob("brats_tta/__init__.py"))
        for init_file in package_inits:
            package_root = init_file.parent.parent
            config_candidates = [
                package_root.parent / "configs" / SOURCE_CONFIG_NAME,
                package_root.parent / SOURCE_CONFIG_NAME,
                init_file.parents[2] / "configs" / SOURCE_CONFIG_NAME,
            ]
            for config_path in config_candidates:
                if config_path.is_file():
                    return package_root, config_path

    # A source dependency may have been uploaded as a single ZIP file.
    for archive in sorted(Path("/kaggle/input").rglob("*.zip")):
        if "source" not in archive.name.lower() and "cicek" not in archive.name.lower():
            continue
        extraction_root = Path("/kaggle/working/cicek_source")
        marker = extraction_root / ".complete"
        if not marker.is_file():
            extraction_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.infolist():
                    destination = (extraction_root / member.filename).resolve()
                    if extraction_root.resolve() not in destination.parents:
                        raise RuntimeError(f"unsafe source archive member: {member.filename}")
                    zipped.extract(member, extraction_root)
            marker.write_text("ok", encoding="utf-8")
        for init_file in sorted(extraction_root.rglob("brats_tta/__init__.py")):
            package_root = init_file.parent.parent
            config_path = package_root.parent / "configs" / SOURCE_CONFIG_NAME
            if config_path.is_file():
                return package_root, config_path

    raise FileNotFoundError("current brats_tta source package or Cicek config was not mounted")


def configure_environment(source_root: Path) -> None:
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    os.environ.setdefault("TORCH_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def build_probe_optimizer(model: Any, training_config: dict[str, Any]) -> Any:
    import torch

    optimizer_name = str(training_config.get("optimizer", "sgd")).lower()
    learning_rate = float(training_config.get("learning_rate", 0.01))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name != "sgd":
        raise ValueError(f"unsupported probe optimizer: {optimizer_name}")
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=float(training_config.get("momentum", 0.0)),
        weight_decay=weight_decay,
        nesterov=bool(training_config.get("nesterov", False)),
    )


def probe_child(
    source_root: Path, config_path: Path, patch_size: tuple[int, int, int], batch_size: int
) -> None:
    """Run one isolated FP32 step so an OOM cannot poison later probes."""

    configure_environment(source_root)
    import torch

    from brats_tta.config import load_config
    from brats_tta.losses.segmentation import build_loss
    from brats_tta.models import build_source_model

    torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
    try:
        torch.set_num_interop_threads(int(os.environ["TORCH_INTEROP_THREADS"]))
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        print(json.dumps({"status": "no_cuda"}), flush=True)
        return

    device = torch.device("cuda:0")
    config = load_config(config_path)
    model = None
    criterion = None
    optimizer = None
    try:
        model = build_source_model(config["model"]).to(device).train()
        criterion = build_loss(
            config["loss"],
            number_of_outputs=1,
            output_mode=config["model"].get("output_mode", "classes_softmax"),
        ).to(device)
        optimizer = build_probe_optimizer(model, config["training"])
        image = torch.randn((batch_size, 4, *patch_size), device=device, dtype=torch.float32)
        target = torch.randint(0, 4, (batch_size, *patch_size), device=device, dtype=torch.long)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        result = {
            "status": "ok",
            "patch_size": list(patch_size),
            "batch_size": batch_size,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "seconds": time.perf_counter() - started,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
            "loss": float(loss.detach().item()),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
    except torch.cuda.OutOfMemoryError as error:
        print(
            json.dumps(
                {
                    "status": "oom",
                    "patch_size": list(patch_size),
                    "batch_size": batch_size,
                    "error": str(error).splitlines()[0][:500],
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print(
                json.dumps(
                    {
                        "status": "oom",
                        "patch_size": list(patch_size),
                        "batch_size": batch_size,
                        "error": str(error).splitlines()[0][:500],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            raise
    finally:
        del optimizer, criterion, model
        gc.collect()
        torch.cuda.empty_cache()


def run_probe(
    source_root: Path, config_path: Path, patch_size: tuple[int, int, int], batch_size: int
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--source-root",
        str(source_root),
        "--config",
        str(config_path),
        "--patch-size",
        *(str(value) for value in patch_size),
        "--batch-size",
        str(batch_size),
    ]
    timeout_seconds = int(os.environ.get("BRATS_PROBE_TIMEOUT_SECONDS", "180"))
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "patch_size": list(patch_size),
            "batch_size": batch_size,
            "timeout_seconds": timeout_seconds,
            "stdout_tail": (error.stdout or "")[-1000:],
            "stderr_tail": (error.stderr or "")[-1000:],
        }
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    json_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if json_lines:
        try:
            result = json.loads(json_lines[-1])
            result["return_code"] = completed.returncode
            return result
        except json.JSONDecodeError:
            pass
    return {
        "status": "error",
        "patch_size": list(patch_size),
        "batch_size": batch_size,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def locate_dataset() -> Path | None:
    input_root = Path("/kaggle/input")
    for directory in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if any(directory.rglob("*-seg.nii")) or any(directory.rglob("*-seg.nii.gz")):
            return directory
    return None


def extract_dataset_if_needed() -> Path | None:
    directory = locate_dataset()
    if directory is not None:
        return directory
    archives = sorted(Path("/kaggle/input").rglob("*.zip"))
    if not archives:
        return None
    archive = next(
        (path for path in archives if "brats" in path.name.lower() and "gli" in path.name.lower()),
        archives[0],
    )
    extraction_root = Path("/kaggle/working/gli_raw")
    marker = extraction_root / ".complete"
    if not marker.is_file():
        extraction_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                destination = (extraction_root / member.filename).resolve()
                if extraction_root.resolve() not in destination.parents:
                    raise RuntimeError(f"unsafe dataset archive member: {member.filename}")
                zipped.extract(member, extraction_root)
        marker.write_text("ok", encoding="utf-8")
    directories = sorted(path for path in extraction_root.iterdir() if path.is_dir())
    return directories[0] if len(directories) == 1 else extraction_root


def run_real_pilot(source_root: Path, config_path: Path) -> dict[str, Any]:
    """Run one direct real-NIfTI training step without importing evaluation code."""

    try:
        dataset_root = extract_dataset_if_needed()
        if dataset_root is None:
            return {"status": "skipped", "reason": "GLI dataset was not mounted"}

        configure_environment(source_root)
        import torch
        from torch.utils.data import DataLoader

        from brats_tta.config import load_config
        from brats_tta.data.brats import BraTSDataset
        from brats_tta.data.manifest import discover_brats_cases, split_cases, write_manifest
        from brats_tta.losses.segmentation import build_loss
        from brats_tta.models import build_source_model

        cases = discover_brats_cases(dataset_root, require_label=True, skip_incomplete=True)
        train_cases, val_cases = split_cases(
            cases,
            validation_fraction=0.2,
            seed=2025,
            group_by_patient=True,
        )
        if not train_cases or not val_cases:
            return {"status": "skipped", "reason": "not enough labeled cases"}
        root = Path("/kaggle/working/cicek_real_pilot")
        train_manifest = root / "train.json"
        write_manifest(train_cases[:2], train_manifest, metadata={"label_schema": "brats_modern"})
        config = load_config(config_path)
        patch_size = (64, 64, 64)
        output_mode = config["model"].get("output_mode", "classes_softmax")
        dataset = BraTSDataset(
            train_manifest,
            training=True,
            patch_size=patch_size,
            label_schema="brats_modern",
            output_mode=output_mode,
            augmentation_config=config["data"].get("augmentation", {}),
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        device = torch.device("cuda:0")
        model = build_source_model(config["model"]).to(device).train()
        criterion = build_loss(config["loss"], 1, output_mode=output_mode).to(device)
        optimizer = build_probe_optimizer(model, config["training"])
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        return {
            "status": "ok",
            "seconds": time.perf_counter() - started,
            "train_cases": len(train_cases[:2]),
            "patch_size": list(patch_size),
            "batch_size": 1,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
            "loss": float(loss.detach().item()),
            "case_id": str(batch["id"][0]),
        }
    except Exception as error:  # noqa: BLE001 - report pilot failures without hiding probe results
        return {
            "status": "error",
            "error": repr(error),
            "traceback": traceback.format_exc()[-4000:],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--patch-size", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if args.child:
        if args.source_root is None or args.config is None:
            raise SystemExit("--source-root and --config are required in child mode")
        probe_child(args.source_root, args.config, tuple(args.patch_size), args.batch_size)
        return

    source_root, config_path = locate_source()
    configure_environment(source_root)
    import torch

    log(f"Started: {utc_now()}")
    log(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    gpu_info = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_info.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_mb": properties.total_memory / (1024**2),
            }
        )
    log(json.dumps({"gpu_info": gpu_info}, sort_keys=True))
    log(f"Source package: {source_root}")
    log(f"Config: {config_path}")

    probes = []
    for patch_size, batch_size in (
        ((64, 64, 64), 1),
        ((96, 96, 96), 1),
        ((128, 128, 128), 1),
        ((128, 128, 128), 2),
    ):
        log(f"Running FP32 probe: patch={patch_size}, batch={batch_size}")
        result = run_probe(source_root, config_path, patch_size, batch_size)
        probes.append(result)
        log(json.dumps(result, sort_keys=True))

    pilot = run_real_pilot(source_root, config_path)
    log("Real-data pilot: " + json.dumps(pilot, sort_keys=True))
    payload: dict[str, Any] = {
        "started_at_utc": utc_now(),
        "source_config": str(config_path),
        "gpu_info": gpu_info,
        "probes": probes,
        "real_data_pilot": pilot,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Results written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
