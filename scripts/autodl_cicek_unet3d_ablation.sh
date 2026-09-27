#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/miniconda3/bin/python" ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="python"
  fi
fi

DATA_ROOT="${DATA_ROOT:-}"
if [[ -z "${DATA_ROOT}" || ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT must point to the mounted GLI directory." >&2
  exit 2
fi

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/brats_tta_cicek_unet3d_ablation_20260923}"
ID_PREFIX="${ID_PREFIX:-BraTS-GLI-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-300}"
ITERATIONS_PER_EPOCH="${ITERATIONS_PER_EPOCH:-250}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VALIDATION_NUM_WORKERS="${VALIDATION_NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
VALIDATION_PREFETCH_FACTOR="${VALIDATION_PREFETCH_FACTOR:-1}"
VALIDATE_EVERY="${VALIDATE_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-20}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SEED="${SEED:-2025}"
RUN_CURRENT_CONTROL="${RUN_CURRENT_CONTROL:-1}"
RUN_NO_BN_ABLATION="${RUN_NO_BN_ABLATION:-1}"
RUN_MEMORY_SMOKE="${RUN_MEMORY_SMOKE:-1}"

CICEK_BN_CONFIG="${CICEK_BN_CONFIG:-${REPO_ROOT}/configs/source_brats_gli_cicek_unet3d_bn.yaml}"
CURRENT_CONFIG="${CURRENT_CONFIG:-${REPO_ROOT}/configs/source_brats_gli_4class_bn.yaml}"
CICEK_NO_BN_CONFIG="${CICEK_NO_BN_CONFIG:-${REPO_ROOT}/configs/source_brats_gli_cicek_unet3d_no_bn.yaml}"
MANIFEST_ROOT="${WORK_ROOT}/manifests"
TRAIN_MANIFEST="${MANIFEST_ROOT}/gli_train_patient.json"
VAL_MANIFEST="${MANIFEST_ROOT}/gli_val_patient.json"

for required in "${CICEK_BN_CONFIG}" "${CURRENT_CONFIG}" "${CICEK_NO_BN_CONFIG}"; do
  [[ -f "${required}" ]] || { echo "Missing config: ${required}" >&2; exit 2; }
done

"${PYTHON_BIN}" - <<'PY'
import importlib
for package in ("torch", "numpy", "nibabel", "yaml", "scipy", "tqdm"):
    importlib.import_module(package)
PY

mkdir -p "${MANIFEST_ROOT}" "${WORK_ROOT}/logs"
if [[ ! -s "${TRAIN_MANIFEST}" || ! -s "${VAL_MANIFEST}" ]]; then
  "${PYTHON_BIN}" -m brats_tta.cli.prepare_manifest \
    --root "${DATA_ROOT}" \
    --train-output "${TRAIN_MANIFEST}" \
    --val-output "${VAL_MANIFEST}" \
    --val-fraction "${VAL_FRACTION}" \
    --seed "${SEED}" \
    --split-unit patient \
    --id-prefix "${ID_PREFIX}" \
    --skip-incomplete \
    --label-schema brats_modern \
    2>&1 | tee "${WORK_ROOT}/logs/manifest.log"
fi

run_variant() {
  local name="$1"
  local config="$2"
  local output="${WORK_ROOT}/${name}/run"
  local log="${WORK_ROOT}/logs/${name}.log"
  mkdir -p "${output}"
  if [[ -f "${output}/checkpoints/last.pt" ]]; then
    echo "${name}: last.pt already exists; keeping the completed run." | tee -a "${log}"
    return 0
  fi
  {
    date -Iseconds
    echo "variant=${name}"
    echo "config=${config}"
    echo "data_root=${DATA_ROOT}"
    echo "work_root=${WORK_ROOT}"
    echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
    echo "git_dirty_files=$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | wc -l || true)"
    echo "batch_size=${BATCH_SIZE}"
    echo "epochs=${EPOCHS}"
    echo "iterations_per_epoch=${ITERATIONS_PER_EPOCH}"
    echo "validate_every=${VALIDATE_EVERY}"
    echo "save_every=${SAVE_EVERY}"
    echo "precision=fp32"
    echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
  } | tee "${log}"
  "${PYTHON_BIN}" -m brats_tta.cli.train_source \
    --config "${config}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --output-dir "${output}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --iterations-per-epoch "${ITERATIONS_PER_EPOCH}" \
    --num-workers "${NUM_WORKERS}" \
    --validate-every "${VALIDATE_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --seed "${SEED}" \
    --device cuda \
    --no-amp \
    --set "data.prefetch_factor=${PREFETCH_FACTOR}" \
    --set "data.validation_prefetch_factor=${VALIDATION_PREFETCH_FACTOR}" \
    --set "data.validation_num_workers=${VALIDATION_NUM_WORKERS}" \
    --set data.label_schema=brats_modern \
    --verbose 2>&1 | tee -a "${log}"
}

smoke_variant() {
  local name="$1"
  local config="$2"
  local log="${WORK_ROOT}/logs/${name}_memory_smoke.log"
  "${PYTHON_BIN}" - "${config}" "${BATCH_SIZE}" <<'PY' 2>&1 | tee "${log}"
import json
import sys
import time

import torch

from brats_tta.config import load_config
from brats_tta.engine.trainer import build_optimizer
from brats_tta.losses.segmentation import build_loss
from brats_tta.models import build_source_model

config = load_config(sys.argv[1])
batch_size = int(sys.argv[2])
patch_size = tuple(int(value) for value in config["data"]["patch_size"])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = not bool(config["experiment"].get("deterministic", False))
torch.set_float32_matmul_precision("highest")
device = torch.device("cuda")
model = build_source_model(config["model"]).to(device).train()
number_of_outputs = (
    len(config["model"]["features"]) - 1
    if config["model"].get("deep_supervision", True)
    else 1
)
criterion = build_loss(
    config["loss"],
    number_of_outputs,
    output_mode=config["model"].get("output_mode", "classes_softmax"),
).to(device)
optimizer = build_optimizer(model, config["training"])
image = torch.randn((batch_size, 4, *patch_size), device=device, dtype=torch.float32)
target = torch.randint(0, 4, (batch_size, *patch_size), device=device)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
torch.cuda.synchronize(device)
started = time.perf_counter()
optimizer.zero_grad(set_to_none=True)
outputs = model(image)
loss = criterion(outputs, target)
loss.backward()
optimizer.step()
torch.cuda.synchronize(device)
print(
    json.dumps(
        {
            "model": config["model"]["name"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "patch_size": patch_size,
            "batch_size": batch_size,
            "precision": "fp32",
            "seconds": time.perf_counter() - started,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
        },
        sort_keys=True,
    )
)
PY
}

# Run the requested paper-style 3D U-Net first so it produces a result even if
# the optional controls are stopped later.
if [[ "${RUN_MEMORY_SMOKE}" == "1" ]]; then
  smoke_variant cicek_bn "${CICEK_BN_CONFIG}"
fi
run_variant cicek_bn "${CICEK_BN_CONFIG}"
if [[ "${RUN_CURRENT_CONTROL}" == "1" ]]; then
  run_variant current_6stage_bn "${CURRENT_CONFIG}"
fi
if [[ "${RUN_NO_BN_ABLATION}" == "1" ]]; then
  run_variant cicek_no_bn "${CICEK_NO_BN_CONFIG}"
fi

echo "Cicek 3D U-Net ablation complete: ${WORK_ROOT}"
