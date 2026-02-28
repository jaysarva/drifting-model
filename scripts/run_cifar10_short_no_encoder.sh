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
TRAIN_SEED="${TRAIN_SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/cifar10_short_no_encoder_${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/eval/cifar10_short_no_encoder_${RUN_TAG}}"

DATASET="cifar10"
CONFIG_PATH="${REPO_ROOT}/configs/cifar10-short-no-encoder.yaml"
RUN_NAME="kernel_short_no_encoder"

LOGGER="wandb"
WANDB_PROJECT="drifting-model"

EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_ALPHA="${EVAL_ALPHA:-1.5}"
EVAL_SEED="${EVAL_SEED:-42}"

TRAIN_OUT="${OUTPUT_ROOT}/${RUN_NAME}/s${TRAIN_SEED}"
CKPT_DIR="${TRAIN_OUT}/${DATASET}"
EVAL_OUT="${EVAL_ROOT}/${RUN_NAME}/s${TRAIN_SEED}"

mkdir -p "${TRAIN_OUT}" "${EVAL_OUT}"

echo "Using python: $(command -v python)"
echo "Run tag: ${RUN_TAG}"
echo "Train output: ${TRAIN_OUT}"
echo "Eval output: ${EVAL_OUT}"
echo

python train.py \
  --dataset "${DATASET}" \
  --config "${CONFIG_PATH}" \
  --output_dir "${TRAIN_OUT}" \
  --seed "${TRAIN_SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --logger "${LOGGER}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${RUN_NAME}-s${TRAIN_SEED}-${RUN_TAG}"

python eval.py \
  --checkpoint_dir "${CKPT_DIR}" \
  --dataset "${DATASET}" \
  --output_dir "${EVAL_OUT}" \
  --csv_path "${EVAL_OUT}/fid_metrics.csv" \
  --num_samples "${EVAL_NUM_SAMPLES}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --alpha "${EVAL_ALPHA}" \
  --seed "${EVAL_SEED}" \
  --skip_existing

# Keep CSV + curve; remove heavy artifacts.
if [[ -d "${EVAL_OUT}/artifacts" ]]; then
  rm -rf "${EVAL_OUT}/artifacts"
  echo "Removed ${EVAL_OUT}/artifacts"
fi

echo
echo "Done."
echo "Eval results kept in: ${EVAL_OUT}"
echo "Training outputs kept in: ${TRAIN_OUT}"
