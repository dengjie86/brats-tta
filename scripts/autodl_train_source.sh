#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The container is limited to 16 CPU cores. One thread per process keeps the
# 12 DataLoader workers from oversubscribing the host while the GPU trains.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
# Structured LOGGER output remains complete in training.log; disabling tqdm
# prevents BrokenPipeError when this script is launched through SSH/nohup.
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
if [[ -z "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT is required and must point to the extracted BraTS 2023 GLI training directory." >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Dataset directory not found: ${DATA_ROOT}" >&2
  exit 2
fi

DEFAULT_WORK_BASE="/root/autodl-tmp"
if [[ ! -d "${DEFAULT_WORK_BASE}" ]]; then
  DEFAULT_WORK_BASE="${REPO_ROOT}/outputs"
fi

WORK_ROOT="${WORK_ROOT:-${DEFAULT_WORK_BASE}/brats2023_gli_4class_bn_31m_6stage_300_fp32}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/source_brats_gli_4class_bn.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-300}"
ITERATIONS_PER_EPOCH="${ITERATIONS_PER_EPOCH:-250}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
VALIDATION_PREFETCH_FACTOR="${VALIDATION_PREFETCH_FACTOR:-1}"
VALIDATION_NUM_WORKERS="${VALIDATION_NUM_WORKERS:-2}"
VALIDATE_EVERY="${VALIDATE_EVERY:-25}"
SAVE_EVERY="${SAVE_EVERY:-20}"
VALIDATION_CASES="${VALIDATION_CASES:-64}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SEED="${SEED:-2025}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Training config not found: ${CONFIG}" >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import importlib

for package in ("torch", "numpy", "nibabel", "yaml", "scipy", "tqdm"):
    importlib.import_module(package)
PY

AVAILABLE_GPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if (( AVAILABLE_GPUS < NUM_GPUS )); then
  echo "Requested ${NUM_GPUS} GPU(s), but PyTorch sees ${AVAILABLE_GPUS}." >&2
  exit 2
fi

mkdir -p "${WORK_ROOT}/manifests" "${WORK_ROOT}/run"
DATA_ROOT="$("${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${DATA_ROOT}")"
WORK_ROOT="$("${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${WORK_ROOT}")"
TRAIN_MANIFEST="${WORK_ROOT}/manifests/gli_train_raw.json"
VAL_MANIFEST="${WORK_ROOT}/manifests/gli_val_raw.json"

if [[ ! -s "${TRAIN_MANIFEST}" || ! -s "${VAL_MANIFEST}" ]]; then
  "${PYTHON_BIN}" -m brats_tta.cli.prepare_manifest \
    --root "${DATA_ROOT}" \
    --train-output "${TRAIN_MANIFEST}" \
    --val-output "${VAL_MANIFEST}" \
    --val-fraction "${VAL_FRACTION}" \
    --seed "${SEED}" \
    --label-schema brats_modern
fi

if (( NUM_GPUS > 1 )); then
  TRAIN_COMMAND=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    "--nproc-per-node=${NUM_GPUS}"
  )
else
  TRAIN_COMMAND=("${PYTHON_BIN}")
fi

TRAIN_COMMAND+=(
  -m brats_tta.cli.train_source
  --config "${CONFIG}"
  --train-manifest "${TRAIN_MANIFEST}"
  --val-manifest "${VAL_MANIFEST}"
  --output-dir "${WORK_ROOT}/run"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --iterations-per-epoch "${ITERATIONS_PER_EPOCH}"
  --num-workers "${NUM_WORKERS}"
  --validate-every "${VALIDATE_EVERY}"
  --save-every "${SAVE_EVERY}"
  --seed "${SEED}"
  --device cuda
  --no-amp
  --set "data.prefetch_factor=${PREFETCH_FACTOR}"
  --set "data.validation_prefetch_factor=${VALIDATION_PREFETCH_FACTOR}"
  --set "data.validation_num_workers=${VALIDATION_NUM_WORKERS}"
  --set data.label_schema=brats_modern
)

if [[ -n "${VALIDATION_CASES}" && "${VALIDATION_CASES}" != "all" ]]; then
  TRAIN_COMMAND+=(--set "training.validation_cases=${VALIDATION_CASES}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
  TRAIN_COMMAND+=(--resume "${RESUME_CHECKPOINT}")
fi

ENVIRONMENT_LOG="${WORK_ROOT}/launch_environment.txt"
{
  date -Iseconds
  echo "repo_root=${REPO_ROOT}"
  echo "data_root=${DATA_ROOT}"
  echo "work_root=${WORK_ROOT}"
  echo "config=${CONFIG}"
  echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
  echo "git_head=$(git -C "${REPO_ROOT}" rev-parse --verify HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_begin"
  git -C "${REPO_ROOT}" status --short 2>/dev/null || true
  echo "git_status_end"
  echo "cpu_quota_expected=16"
  echo "cpu_threads_env=OMP_NUM_THREADS=${OMP_NUM_THREADS},MKL_NUM_THREADS=${MKL_NUM_THREADS},OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS},NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
  echo "torch_threads_env=TORCH_NUM_THREADS=${TORCH_NUM_THREADS},TORCH_INTEROP_THREADS=${TORCH_INTEROP_THREADS}"
  echo "tqdm_disable=${TQDM_DISABLE}"
  echo "num_workers=${NUM_WORKERS}"
  echo "prefetch_factor=${PREFETCH_FACTOR}"
  echo "validation_prefetch_factor=${VALIDATION_PREFETCH_FACTOR}"
  echo "validation_num_workers=${VALIDATION_NUM_WORKERS}"
  echo "memory_limit_expected_gib=80"
  printf 'launch_command='
  printf ' %q' "${TRAIN_COMMAND[@]}"
  printf '\n'
  "${PYTHON_BIN}" -c 'import sys, torch; print(f"python={sys.version.split()[0]}"); print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"gpu_count={torch.cuda.device_count()}"); [print(f"gpu_{i}={torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'
} > "${ENVIRONMENT_LOG}"

"${PYTHON_BIN}" -m pip freeze > "${WORK_ROOT}/pip_freeze.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "${WORK_ROOT}/nvidia_smi.txt" || true
fi

GPU_MONITOR_PID=""
cleanup_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup_gpu_monitor EXIT
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"
  {
    echo "timestamp, index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu"
    while true; do
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits || true
      sleep "${GPU_MONITOR_INTERVAL}"
    done
  } > "${WORK_ROOT}/gpu_utilization.csv" 2>&1 &
  GPU_MONITOR_PID=$!
fi

printf 'Launching:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'
"${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${WORK_ROOT}/console.log"
