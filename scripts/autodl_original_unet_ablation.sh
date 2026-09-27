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

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/brats_tta_original_unet_ablation_20260922}"
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

STRICT_CONFIG="${STRICT_CONFIG:-${REPO_ROOT}/configs/source_brats_gli_original_unet3d.yaml}"
BN_CONFIG="${BN_CONFIG:-${REPO_ROOT}/configs/source_brats_gli_original_unet3d_bn_control.yaml}"
MANIFEST_ROOT="${WORK_ROOT}/manifests"
TRAIN_MANIFEST="${MANIFEST_ROOT}/gli_train_patient.json"
VAL_MANIFEST="${MANIFEST_ROOT}/gli_val_patient.json"

for required in "${STRICT_CONFIG}" "${BN_CONFIG}"; do
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
    echo "${name}: last.pt already exists; set a new WORK_ROOT to start a clean run." | tee -a "${log}"
    return 0
  fi
  {
    date -Iseconds
    echo "variant=${name}"
    echo "config=${config}"
    echo "data_root=${DATA_ROOT}"
    echo "work_root=${WORK_ROOT}"
    echo "id_prefix=${ID_PREFIX}"
    echo "batch_size=${BATCH_SIZE}"
    echo "epochs=${EPOCHS}"
    echo "iterations_per_epoch=${ITERATIONS_PER_EPOCH}"
    echo "validate_every=${VALIDATE_EVERY}"
    echo "save_every=${SAVE_EVERY}"
    echo "precision=fp32"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
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

run_variant strict "${STRICT_CONFIG}"
run_variant bn_control "${BN_CONFIG}"
echo "Original U-Net ablation complete: ${WORK_ROOT}"
