#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/vietanh21/brats-2021-task-1-dataset}"
WORK_ROOT="${WORK_ROOT:-/kaggle/working/brats2021_source_ddp}"
EPOCHS="${EPOCHS:-1000}"
ITERATIONS_PER_EPOCH="${ITERATIONS_PER_EPOCH:-250}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# Kaggle normally exposes a dataset by slug directly below /kaggle/input.
if [[ ! -d "${DATA_ROOT}" && -d "/kaggle/input/brats-2021-task-1-dataset" ]]; then
  DATA_ROOT="/kaggle/input/brats-2021-task-1-dataset"
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Dataset directory not found: ${DATA_ROOT}" >&2
  exit 2
fi

python -c 'import torch; assert torch.cuda.device_count() >= 2, f"Expected two GPUs, found {torch.cuda.device_count()}"; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'

mkdir -p "${WORK_ROOT}/manifests"
TRAIN_MANIFEST="${WORK_ROOT}/manifests/gli_train_raw.json"
VAL_MANIFEST="${WORK_ROOT}/manifests/gli_val_raw.json"

if [[ ! -f "${TRAIN_MANIFEST}" || ! -f "${VAL_MANIFEST}" ]]; then
  python -m brats_tta.cli.prepare_manifest \
    --root "${DATA_ROOT}" \
    --train-output "${TRAIN_MANIFEST}" \
    --val-output "${VAL_MANIFEST}" \
    --val-fraction 0.2 \
    --seed 2025
fi

TRAIN_COMMAND=(
  torchrun
  --standalone
  --nproc-per-node=2
  -m brats_tta.cli.train_source
  --config "${REPO_ROOT}/configs/source_brats_gli.yaml"
  --train-manifest "${TRAIN_MANIFEST}"
  --val-manifest "${VAL_MANIFEST}"
  --output-dir "${WORK_ROOT}/run"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --iterations-per-epoch "${ITERATIONS_PER_EPOCH}"
  --num-workers "${NUM_WORKERS}"
  --device cuda
  --amp
  --set data.label_schema=brats_legacy
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_COMMAND+=(--resume "${RESUME_CHECKPOINT}")
fi

printf 'Launching:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'
"${TRAIN_COMMAND[@]}"
