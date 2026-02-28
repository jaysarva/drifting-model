#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VENV_ACTIVATE="/.venv/bin/activate"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Missing virtual environment activate script: ${VENV_ACTIVATE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_ACTIVATE}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ldf_verification_${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/eval/ldf_verification_${RUN_TAG}}"

DATASET="${DATASET:-cifar10}"
LOGGER="wandb"
WANDB_PROJECT="drifting-model"

EVAL_DATASET="${EVAL_DATASET:-cifar10}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_ALPHA="${EVAL_ALPHA:-1.5}"
EVAL_SEED="${EVAL_SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"

echo "Using python: $(command -v python)"
echo "Run tag: ${RUN_TAG}"
echo "Training output root: ${OUTPUT_ROOT}"
echo "Eval output root: ${EVAL_ROOT}"
echo

run_train_eval() {
  local run_name="$1"
  local config_rel="$2"
  local seed="$3"

  local config_path="${REPO_ROOT}/${config_rel}"
  local train_out="${OUTPUT_ROOT}/${run_name}/s${seed}"
  local eval_out="${EVAL_ROOT}/${run_name}/s${seed}"
  local ckpt_dir="${train_out}/${DATASET}"

  mkdir -p "${train_out}" "${eval_out}"

  echo "================================================================="
  echo "Run: ${run_name} | Seed: ${seed}"
  echo "Config: ${config_rel}"
  echo "================================================================="

  local -a train_cmd=(
    python train.py
    --dataset "${DATASET}"
    --config "${config_path}"
    --output_dir "${train_out}"
    --seed "${seed}"
    --logger "${LOGGER}"
    --num_workers "${NUM_WORKERS}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_run_name "${run_name}-s${seed}-${RUN_TAG}"
  )

  "${train_cmd[@]}"

  local -a eval_cmd=(
    python eval.py
    --checkpoint_dir "${ckpt_dir}"
    --dataset "${EVAL_DATASET}"
    --output_dir "${eval_out}"
    --csv_path "${eval_out}/fid_metrics.csv"
    --num_samples "${EVAL_NUM_SAMPLES}"
    --batch_size "${EVAL_BATCH_SIZE}"
    --alpha "${EVAL_ALPHA}"
    --seed "${EVAL_SEED}"
    --skip_existing
  )

  "${eval_cmd[@]}"

  # Keep CSV + curve; remove heavy per-checkpoint artifacts after eval is complete.
  if [[ -d "${eval_out}/artifacts" ]]; then
    rm -rf "${eval_out}/artifacts"
    echo "Removed ${eval_out}/artifacts"
  fi

  # Eval is complete for this run; remove corresponding training outputs/checkpoints.
  if [[ -d "${train_out}" ]]; then
    rm -rf "${train_out}"
    echo "Removed ${train_out}"
  fi

  echo
}

echo "Starting LDF verification matrix..."
echo

# Baseline is assumed to be pre-existing from configs/cifar10-short.yaml.
# Run only short-budget LDF variants for direct comparison.
run_train_eval "ldf_short" "configs/cifar10-ldf.yaml" "${TRAIN_SEED}"
run_train_eval "ldf_no_proj_short" "configs/cifar10-ldf-no-proj.yaml" "${TRAIN_SEED}"
run_train_eval "ldf_set_small_short" "configs/cifar10-ldf-setsize-small.yaml" "${TRAIN_SEED}"
run_train_eval "ldf_set_large_short" "configs/cifar10-ldf-setsize-large.yaml" "${TRAIN_SEED}"

echo "All runs complete."
echo "Training outputs: ${OUTPUT_ROOT}"
echo "Eval outputs: ${EVAL_ROOT}"
