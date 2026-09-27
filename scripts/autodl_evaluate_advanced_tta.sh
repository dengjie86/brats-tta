#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/miniconda3/bin/python" ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="python"
  fi
fi

CHECKPOINT="${CHECKPOINT:-}"
SSA_MANIFEST="${SSA_MANIFEST:-}"
PED_MANIFEST="${PED_MANIFEST:-}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-fs/brats_advanced_tta_full_20260921}"
PATCH_SIZE="${PATCH_SIZE:-128}"
OVERLAP="${OVERLAP:-0.5}"
TEGDA_LR="${TEGDA_LR:-0.0001}"
TEGDA_SCOPE="${TEGDA_SCOPE:-full}"
TEGDA_DROPOUT_SAMPLES="${TEGDA_DROPOUT_SAMPLES:-10}"
TEGDA_DROPOUT_PROBABILITY="${TEGDA_DROPOUT_PROBABILITY:-0.5}"
TEGDA_FEATURE_BANK_SIZE="${TEGDA_FEATURE_BANK_SIZE:-10}"
TEGDA_FEATURE_PERCENTILE="${TEGDA_FEATURE_PERCENTILE:-0.90}"
RUN_SAR="${RUN_SAR:-1}"
RUN_SAR_NO_RECOVERY="${RUN_SAR_NO_RECOVERY:-1}"
RUN_VPTTA="${RUN_VPTTA:-1}"
RUN_TEGDA="${RUN_TEGDA:-1}"
RUN_TEGDA_SMOKE="${RUN_TEGDA_SMOKE:-1}"

for variable_name in CHECKPOINT SSA_MANIFEST PED_MANIFEST; do
  value="${!variable_name}"
  if [[ -z "${value}" || ! -f "${value}" ]]; then
    echo "${variable_name} must point to an existing file; got: ${value:-<empty>}" >&2
    exit 2
  fi
done
if [[ "${TEGDA_SCOPE}" != "norm" && "${TEGDA_SCOPE}" != "full" ]]; then
  echo "TEGDA_SCOPE must be norm or full, got: ${TEGDA_SCOPE}" >&2
  exit 2
fi

mkdir -p "${WORK_ROOT}/logs"

run_method() {
  local variant="$1"
  local method="$2"
  local dataset="$3"
  local manifest="$4"
  shift 4

  local output_dir="${WORK_ROOT}/${variant}/${dataset}"
  local log_path="${WORK_ROOT}/logs/${variant}_${dataset}.log"
  local command=(
    "${PYTHON_BIN}" -m brats_tta.cli.evaluate_advanced_tta
    --checkpoint "${CHECKPOINT}"
    --manifest "${manifest}"
    --output-dir "${output_dir}"
    --method "${method}"
    --device cuda
    --patch-size "${PATCH_SIZE}" "${PATCH_SIZE}" "${PATCH_SIZE}"
    --overlap "${OVERLAP}"
    --sw-batch-size 1
    --hd95
    --stall-trace-seconds 300
    "$@"
  )

  printf '%s starting %s/%s\n' "$(date -Iseconds)" "${variant}" "${dataset}" |
    tee -a "${log_path}"
  printf 'command=' | tee -a "${log_path}"
  printf ' %q' "${command[@]}" | tee -a "${log_path}"
  printf '\n' | tee -a "${log_path}"
  "${command[@]}" 2>&1 | tee -a "${log_path}"
}

run_tegda_smoke() {
  local output_dir="${WORK_ROOT}/smoke/tegda_ssa"
  local log_path="${WORK_ROOT}/logs/tegda_smoke_ssa.log"
  local command=(
    "${PYTHON_BIN}" -m brats_tta.cli.evaluate_advanced_tta
    --checkpoint "${CHECKPOINT}"
    --manifest "${SSA_MANIFEST}"
    --output-dir "${output_dir}"
    --method tegda
    --device cuda
    --patch-size "${PATCH_SIZE}" "${PATCH_SIZE}" "${PATCH_SIZE}"
    --overlap "${OVERLAP}"
    --sw-batch-size 1
    --hd95
    --limit 1
    --overwrite
    --stall-trace-seconds 300
    --tegda-lr "${TEGDA_LR}"
    --tegda-scope "${TEGDA_SCOPE}"
    --tegda-dropout-samples "${TEGDA_DROPOUT_SAMPLES}"
    --tegda-dropout-probability "${TEGDA_DROPOUT_PROBABILITY}"
    --tegda-feature-bank-size "${TEGDA_FEATURE_BANK_SIZE}"
    --tegda-feature-percentile "${TEGDA_FEATURE_PERCENTILE}"
  )

  printf '%s starting tegda/smoke_ssa\n' "$(date -Iseconds)" | tee -a "${log_path}"
  printf 'command=' | tee -a "${log_path}"
  printf ' %q' "${command[@]}" | tee -a "${log_path}"
  printf '\n' | tee -a "${log_path}"
  "${command[@]}" 2>&1 | tee -a "${log_path}"
}

{
  date -Iseconds
  echo "repo_root=${REPO_ROOT}"
  echo "work_root=${WORK_ROOT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "checkpoint_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
  echo "ssa_manifest=${SSA_MANIFEST}"
  echo "ssa_manifest_sha256=$(sha256sum "${SSA_MANIFEST}" | awk '{print $1}')"
  echo "ped_manifest=${PED_MANIFEST}"
  echo "ped_manifest_sha256=$(sha256sum "${PED_MANIFEST}" | awk '{print $1}')"
  echo "patch_size=${PATCH_SIZE}"
  echo "overlap=${OVERLAP}"
  echo "tegda_lr=${TEGDA_LR}"
  echo "tegda_scope=${TEGDA_SCOPE}"
  echo "tegda_dropout_samples=${TEGDA_DROPOUT_SAMPLES}"
  echo "tegda_dropout_probability=${TEGDA_DROPOUT_PROBABILITY}"
  echo "tegda_feature_bank_size=${TEGDA_FEATURE_BANK_SIZE}"
  echo "tegda_feature_percentile=${TEGDA_FEATURE_PERCENTILE}"
  echo "volume_protocol=native_grid_sliding_window"
  echo "adaptation_unit=sliding_window_patch_batch"
  echo "tegda_reference_protocol=whole_volume_resized_128_cubed"
  echo "git_head=$(git -C "${REPO_ROOT}" rev-parse --verify HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_begin"
  git -C "${REPO_ROOT}" status --short 2>/dev/null || true
  echo "git_status_end"
  "${PYTHON_BIN}" -c 'import sys, torch; print(f"python={sys.version.split()[0]}"); print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"gpu_count={torch.cuda.device_count()}"); [print(f"gpu_{i}={torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'
} > "${WORK_ROOT}/launch_environment.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "${WORK_ROOT}/nvidia_smi.txt" || true
fi
"${PYTHON_BIN}" -m pip freeze > "${WORK_ROOT}/pip_freeze.txt"

if [[ "${RUN_TEGDA}" == "1" && "${RUN_TEGDA_SMOKE}" == "1" ]]; then
  run_tegda_smoke
fi

if [[ "${RUN_SAR}" == "1" ]]; then
  for dataset in ssa ped; do
    if [[ "${dataset}" == "ssa" ]]; then
      manifest="${SSA_MANIFEST}"
    else
      manifest="${PED_MANIFEST}"
    fi
    run_method sar sar "${dataset}" "${manifest}"
  done
fi

if [[ "${RUN_SAR_NO_RECOVERY}" == "1" ]]; then
  for dataset in ssa ped; do
    if [[ "${dataset}" == "ssa" ]]; then
      manifest="${SSA_MANIFEST}"
    else
      manifest="${PED_MANIFEST}"
    fi
    run_method sar_no_recovery sar "${dataset}" "${manifest}" --sar-no-recovery
  done
fi

if [[ "${RUN_VPTTA}" == "1" ]]; then
  for dataset in ssa ped; do
    if [[ "${dataset}" == "ssa" ]]; then
      manifest="${SSA_MANIFEST}"
    else
      manifest="${PED_MANIFEST}"
    fi
    run_method vptta vptta "${dataset}" "${manifest}"
  done
fi

if [[ "${RUN_TEGDA}" == "1" ]]; then
  for dataset in ssa ped; do
    if [[ "${dataset}" == "ssa" ]]; then
      manifest="${SSA_MANIFEST}"
    else
      manifest="${PED_MANIFEST}"
    fi
    run_method tegda tegda "${dataset}" "${manifest}" \
      --tegda-lr "${TEGDA_LR}" \
      --tegda-scope "${TEGDA_SCOPE}" \
      --tegda-dropout-samples "${TEGDA_DROPOUT_SAMPLES}" \
      --tegda-dropout-probability "${TEGDA_DROPOUT_PROBABILITY}" \
      --tegda-feature-bank-size "${TEGDA_FEATURE_BANK_SIZE}" \
      --tegda-feature-percentile "${TEGDA_FEATURE_PERCENTILE}"
  done
fi
