from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

KERNEL_ROOT = Path(__file__).resolve().parent
SOURCE_CONFIG_NAME = "source_brats_gli_cicek_unet3d_bn.yaml"
WORK_ROOT = Path("/kaggle/working/brats_gli_cicek_unet3d_fp32")
TEMP_ROOT = Path("/kaggle/temp/brats_gli_cicek_unet3d_fp32")
RAW_ROOT = TEMP_ROOT / "raw"
MANIFEST_ROOT = TEMP_ROOT / "manifests"
OUTPUT_MANIFEST_ROOT = WORK_ROOT / "manifests"
RUN_ROOT = WORK_ROOT / "run"
SUMMARY_PATH = WORK_ROOT / "training_job_summary.json"
WRAPPER_LOG_PATH = WORK_ROOT / "training_wrapper.log"
SEGMENT_EPOCHS = int(os.environ.get("BRATS_SEGMENT_EPOCHS", "20"))
GLOBAL_TARGET_EPOCHS = int(os.environ.get("BRATS_GLOBAL_TARGET_EPOCHS", "300"))
ITERATIONS_PER_EPOCH = int(os.environ.get("BRATS_ITERATIONS_PER_EPOCH", "250"))
BATCH_SIZE = 1
NUM_WORKERS = int(os.environ.get("BRATS_NUM_WORKERS", "2"))
VALIDATION_NUM_WORKERS = int(os.environ.get("BRATS_VALIDATION_NUM_WORKERS", "1"))
VALIDATE_EVERY = int(os.environ.get("BRATS_VALIDATE_EVERY", "20"))
SAVE_EVERY = int(os.environ.get("BRATS_SAVE_EVERY", "20"))
MAX_RUNTIME_SECONDS = int(os.environ.get("BRATS_MAX_RUNTIME_SECONDS", "39000"))
STOP_REQUESTED = False
RESUME_SOURCE: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


def _safe_extract(archive: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            destination = (destination_root / member.filename).resolve()
            destination_root_resolved = destination_root.resolve()
            if (
                destination_root_resolved not in destination.parents
                and destination != destination_root_resolved
            ):
                raise RuntimeError(f"unsafe archive member: {member.filename}")
            zipped.extract(member, destination_root)


def _find_source_package() -> tuple[Path, Path]:
    candidates = [KERNEL_ROOT, Path("/kaggle/working"), Path("/kaggle/input")]
    for root in candidates:
        if not root.is_dir():
            continue
        for init_file in sorted(root.rglob("brats_tta/__init__.py")):
            source_package = init_file.parent.parent
            config_candidates = (
                source_package.parent.parent / "configs" / SOURCE_CONFIG_NAME,
                source_package.parent / "configs" / SOURCE_CONFIG_NAME,
                init_file.parents[2] / "configs" / SOURCE_CONFIG_NAME,
            )
            for config_path in config_candidates:
                if config_path.is_file():
                    return source_package, config_path

    extraction_root = Path("/kaggle/temp/cicek_source")
    marker = extraction_root / ".complete"
    if not marker.is_file():
        archives = sorted(
            path for path in Path("/kaggle/input").rglob("*.zip") if path.is_file()
        )
        for archive in archives:
            if archive.stat().st_size > 2 * 1024**3:
                continue
            with zipfile.ZipFile(archive) as zipped:
                names = {Path(name).as_posix() for name in zipped.namelist()}
            if any(name.endswith("brats_tta/__init__.py") for name in names):
                _safe_extract(archive, extraction_root)
            if any(name.endswith(SOURCE_CONFIG_NAME) for name in names):
                _safe_extract(archive, extraction_root)
        marker.write_text("ok", encoding="utf-8")
    for init_file in sorted(extraction_root.rglob("brats_tta/__init__.py")):
        source_package = init_file.parent.parent
        config_path = source_package.parent / "configs" / SOURCE_CONFIG_NAME
        if config_path.is_file():
            return source_package, config_path
    raise FileNotFoundError("the current brats_tta source package and Cicek config were not mounted")


def _locate_gli_archive() -> Path | None:
    preferred_name = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip".lower()
    archives = sorted(path for path in Path("/kaggle/input").rglob("*.zip") if path.is_file())
    preferred = [path for path in archives if path.name.lower() == preferred_name]
    return preferred[0] if preferred else next(
        (path for path in archives if "brats" in path.name.lower() and "gli" in path.name.lower()),
        None,
    )


def _locate_expanded_dataset() -> Path:
    input_root = Path("/kaggle/input")
    for candidate in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if any(candidate.rglob("*-seg.nii")) or any(candidate.rglob("*-seg.nii.gz")):
            return candidate
    raise FileNotFoundError("no BraTS GLI ZIP or expanded NIfTI directory was found below /kaggle/input")


def resolve_dataset_root() -> Path:
    marker = RAW_ROOT / ".extraction_complete.json"
    if marker.is_file():
        extracted_root = Path(json.loads(marker.read_text(encoding="utf-8"))["extracted_root"])
        if extracted_root.is_dir():
            log(f"Reusing extracted GLI dataset: {extracted_root}")
            return extracted_root

    archive = _locate_gli_archive()
    if archive is None:
        dataset_root = _locate_expanded_dataset()
        log(f"Using expanded GLI dataset: {dataset_root}")
        return dataset_root

    log(f"Extracting GLI archive ({archive.stat().st_size / 1024**3:.2f} GiB): {archive}")
    _safe_extract(archive, RAW_ROOT)
    top_level_directories = sorted(path for path in RAW_ROOT.iterdir() if path.is_dir())
    extracted_root = top_level_directories[0] if len(top_level_directories) == 1 else RAW_ROOT
    marker.write_text(
        json.dumps(
            {
                "archive": str(archive),
                "extracted_root": str(extracted_root),
                "completed_at_utc": utc_now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return extracted_root


def _find_resume_artifacts() -> Path | None:
    input_root = Path("/kaggle/input")
    latest_paths = sorted(
        path for path in input_root.rglob("latest.pt") if path.is_file() and path.parent.name == "checkpoints"
    )
    if not latest_paths:
        extraction_root = Path("/kaggle/temp/cicek_resume")
        extraction_root.mkdir(parents=True, exist_ok=True)
        for archive in sorted(input_root.rglob("*.zip")):
            if archive.stat().st_size > 8 * 1024**3:
                continue
            try:
                with zipfile.ZipFile(archive) as zipped:
                    names = {Path(name).as_posix() for name in zipped.namelist()}
            except zipfile.BadZipFile:
                continue
            if any(
                Path(name).name == "latest.pt" and "checkpoints" in Path(name).parts
                for name in names
            ):
                _safe_extract(archive, extraction_root)
        latest_paths = sorted(
            path
            for path in extraction_root.rglob("latest.pt")
            if path.is_file() and path.parent.name == "checkpoints"
        )
    if not latest_paths:
        return None
    latest = latest_paths[0]
    resume_root = latest.parent.parent
    if not (resume_root / "checkpoints" / "latest.pt").is_file():
        raise FileNotFoundError(f"resume checkpoint directory is incomplete: {resume_root}")
    return resume_root


def restore_resume_artifacts() -> tuple[str | None, int]:
    global RESUME_SOURCE
    resume_root = _find_resume_artifacts()
    if resume_root is None:
        return None, 0
    shutil.copytree(resume_root, RUN_ROOT, dirs_exist_ok=True)
    latest = RUN_ROOT / "checkpoints" / "latest.pt"
    checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
    completed_epoch = int(checkpoint["epoch"]) + 1
    RESUME_SOURCE = str(resume_root)
    log(f"Restored resume artifacts from {resume_root}; completed_epoch={completed_epoch}")
    return str(latest), completed_epoch


def prepare_manifests(dataset_root: Path) -> tuple[Path, Path, int, int]:
    train_manifest = MANIFEST_ROOT / "gli_train_patient.json"
    val_manifest = MANIFEST_ROOT / "gli_val_patient.json"
    if not train_manifest.is_file() or not val_manifest.is_file():
        from brats_tta.data.manifest import discover_brats_cases, split_cases, write_manifest

        cases = discover_brats_cases(dataset_root, require_label=True, skip_incomplete=False)
        train_cases, val_cases = split_cases(
            cases,
            validation_fraction=0.2,
            seed=2025,
            group_by_patient=True,
        )
        metadata = {
            "dataset_root": str(dataset_root),
            "label_schema": "brats_modern",
            "split_seed": 2025,
            "split_unit": "patient",
        }
        write_manifest(train_cases, train_manifest, metadata={**metadata, "split": "train"})
        write_manifest(val_cases, val_manifest, metadata={**metadata, "split": "validation"})
        log(
            "Prepared patient-level manifests: "
            f"total={len(cases)}, train={len(train_cases)}, validation={len(val_cases)}"
        )
    train_count = len(json.loads(train_manifest.read_text(encoding="utf-8"))["cases"])
    val_count = len(json.loads(val_manifest.read_text(encoding="utf-8"))["cases"])
    OUTPUT_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_manifest, OUTPUT_MANIFEST_ROOT / train_manifest.name)
    shutil.copy2(val_manifest, OUTPUT_MANIFEST_ROOT / val_manifest.name)
    return train_manifest, val_manifest, train_count, val_count


def write_runtime_config(
    source_config: Path,
    train_manifest: Path,
    val_manifest: Path,
    target_epochs: int,
) -> Path:
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["experiment"]["output_dir"] = str(RUN_ROOT)
    config["data"]["train_manifest"] = str(train_manifest)
    config["data"]["val_manifest"] = str(val_manifest)
    config["data"]["label_schema"] = "brats_modern"
    config["data"]["num_workers"] = NUM_WORKERS
    config["data"]["validation_num_workers"] = VALIDATION_NUM_WORKERS
    config["data"]["prefetch_factor"] = 2
    config["data"]["validation_prefetch_factor"] = 1
    config["training"]["batch_size"] = BATCH_SIZE
    config["training"]["epochs"] = target_epochs
    config["training"]["scheduler_epochs"] = GLOBAL_TARGET_EPOCHS
    config["training"]["iterations_per_epoch"] = ITERATIONS_PER_EPOCH
    config["training"]["validate_every"] = VALIDATE_EVERY
    config["training"]["save_every"] = SAVE_EVERY
    config["training"]["validation_cases"] = None
    config["training"]["amp"] = False
    config["training"]["log_every"] = 1
    config["training"]["validation_log_every"] = 1
    config["model"]["sync_batchnorm"] = True
    config["inference"]["amp"] = False
    runtime_config = WORK_ROOT / "source_brats_gli_cicek_unet3d_bn_runtime.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return runtime_config


def build_training_command(runtime_config: Path, resume_checkpoint: str | None) -> list[str]:
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 1:
        raise RuntimeError("the Kaggle kernel has no visible CUDA GPU")
    base = [
        "-m",
        "brats_tta.cli.train_source",
        "--config",
        str(runtime_config),
        "--batch-size",
        str(BATCH_SIZE),
        "--epochs",
        str(int(yaml.safe_load(runtime_config.read_text(encoding="utf-8"))["training"]["epochs"])),
        "--iterations-per-epoch",
        str(ITERATIONS_PER_EPOCH),
        "--num-workers",
        str(NUM_WORKERS),
        "--device",
        "cuda",
        "--no-amp",
        "--set",
        "data.validation_num_workers=1",
        "--set",
        "data.prefetch_factor=2",
        "--set",
        "data.validation_prefetch_factor=1",
        "--set",
        "model.sync_batchnorm=true",
        "--verbose",
    ]
    if resume_checkpoint:
        base.extend(["--resume", resume_checkpoint])
    if visible_gpus >= 2:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={visible_gpus}",
            *base,
        ]
    return [sys.executable, *base]


def request_stop(process: subprocess.Popen[str]) -> None:
    global STOP_REQUESTED
    if process.poll() is not None:
        return
    STOP_REQUESTED = True
    log("Runtime budget reached; requesting a graceful training stop")
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass


def stream_training(process: subprocess.Popen[str], started_monotonic: float) -> int:
    deadline = started_monotonic + MAX_RUNTIME_SECONDS if MAX_RUNTIME_SECONDS > 0 else None

    def watchdog() -> None:
        if deadline is None:
            return
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        request_stop(process)

    threading.Thread(target=watchdog, daemon=True).start()
    with WRAPPER_LOG_PATH.open("a", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
    return process.wait()


def checkpoint_metadata() -> dict[str, Any]:
    latest = RUN_ROOT / "checkpoints" / "latest.pt"
    if not latest.is_file():
        return {"latest_checkpoint": None}
    try:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        return {
            "latest_checkpoint": str(latest),
            "completed_epoch": int(checkpoint["epoch"]) + 1,
            "best_dice": float(checkpoint.get("best_dice", float("nan"))),
        }
    except Exception as error:  # pragma: no cover - defensive reporting path
        return {"latest_checkpoint": str(latest), "checkpoint_read_error": repr(error)}


def write_summary(
    *,
    status: str,
    started_at: str,
    finished_at: str,
    return_code: int | None,
    train_count: int,
    val_count: int,
    target_epochs: int,
    command: list[str] | None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "return_code": return_code,
        "global_target_epochs": GLOBAL_TARGET_EPOCHS,
        "segment_epochs": SEGMENT_EPOCHS,
        "target_epochs_this_run": target_epochs,
        "iterations_per_epoch": ITERATIONS_PER_EPOCH,
        "batch_size_per_gpu": BATCH_SIZE,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "train_cases": train_count,
        "validation_cases": val_count,
        "validate_every": VALIDATE_EVERY,
        "save_every": SAVE_EVERY,
        "precision": "strict_fp32",
        "sync_batchnorm": True,
        "resume_source": RESUME_SOURCE,
        "command": command,
        **checkpoint_metadata(),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    log(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    global STOP_REQUESTED
    started_at = utc_now()
    started_monotonic = time.monotonic()
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TORCH_NUM_THREADS"] = "1"
    os.environ["TORCH_INTEROP_THREADS"] = "1"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

    source_package, source_config = _find_source_package()
    sys.path.insert(0, str(source_package))
    os.environ["PYTHONPATH"] = str(source_package) + os.pathsep + os.environ.get("PYTHONPATH", "")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"Python: {sys.version}")
    log(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    log(f"Visible GPUs: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        log(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    log(f"Source package: {source_package}")
    log(f"Source config: {source_config}")

    resume_checkpoint, previous_completed_epoch = restore_resume_artifacts()
    if os.environ.get("BRATS_TARGET_EPOCHS"):
        target_epochs = int(os.environ["BRATS_TARGET_EPOCHS"])
    else:
        target_epochs = min(GLOBAL_TARGET_EPOCHS, previous_completed_epoch + SEGMENT_EPOCHS)

    dataset_root = resolve_dataset_root()
    train_manifest, val_manifest, train_count, val_count = prepare_manifests(dataset_root)
    runtime_config = write_runtime_config(source_config, train_manifest, val_manifest, target_epochs)
    command = build_training_command(runtime_config, resume_checkpoint)
    log(
        f"Training target: completed epochs {previous_completed_epoch}->{target_epochs} "
        f"/ {GLOBAL_TARGET_EPOCHS}; "
        f"iterations/epoch={ITERATIONS_PER_EPOCH}, batch/GPU={BATCH_SIZE}, validation=full"
    )
    log("Launching training command: " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(source_package.parent.parent),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    return_code = stream_training(process, started_monotonic)
    if STOP_REQUESTED:
        status = "stopped_for_time_budget"
    elif return_code == 0:
        status = "training_complete"
    else:
        status = "training_failed"
    write_summary(
        status=status,
        started_at=started_at,
        finished_at=utc_now(),
        return_code=return_code,
        train_count=train_count,
        val_count=val_count,
        target_epochs=target_epochs,
        command=command,
    )
    if status == "training_failed":
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
