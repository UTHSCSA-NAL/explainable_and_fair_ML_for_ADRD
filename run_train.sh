#!/usr/bin/env bash
set -euo pipefail

# Activate conda (adjust if your conda.sh is elsewhere)
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  set +u; source "${CONDA_SH}"; set -u
else
  set +u; source "${HOME}/.bashrc" || true; set -u
fi
CONDA_ENV="${CONDA_ENV:-dementia_ai}"
conda activate "${CONDA_ENV}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (override via env)
TRAIN_CSV="${TRAIN_CSV:-${ROOT_DIR}/data_splits/train.csv}"
TEST_CSV="${TEST_CSV:-${ROOT_DIR}/data_splits/test.csv}"
EXTERNAL_CSV="${EXTERNAL_CSV:-${ROOT_DIR}/data_splits/external.csv}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT_DIR}/runs}"
SCENARIO="${SCENARIO:-all}"
MODEL="${MODEL:-xgboost_vanilla}"
METHOD="${METHOD:-baseline}"
LOSS_MODE="${LOSS_MODE:-ce+target+align+gap}"
SCALE_MODE="${SCALE_MODE:-0}"
TEST_SIZE="${TEST_SIZE:-0.2}"
N_REPEATS="${N_REPEATS:-100}"
RANDOM_STATE_BASE="${RANDOM_STATE_BASE:-2025}"
CV_SPLITS="${CV_SPLITS:-5}"
EARLY_STOPPING="${EARLY_STOPPING:-50}"
RUN_RANGE="${RUN_RANGE:-0-9}"
REFIT="${REFIT:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
KMM_WEIGHTS_DIR="${KMM_WEIGHTS_DIR:-${ROOT_DIR}/checkpoints/kmm_weights_small}"
CR_ALPHA_DEFAULT="${CR_ALPHA_DEFAULT:-0.05}"
MERGED_CSV="${MERGED_CSV:-}"

# Expand run range to RNG seed
RUN_START=${RUN_RANGE%-*}
RUN_END=${RUN_RANGE#*-}
if [[ "${RUN_START}" == "${RUN_RANGE}" ]]; then
  RUN_START=${RUN_RANGE}; RUN_END=${RUN_RANGE}
fi

# Expand scenario/method lists (comma/space-separated)
SCENARIO_CLEAN="${SCENARIO//,/ }"
METHOD_CLEAN="${METHOD//,/ }"
read -r -a SCENARIO_LIST <<< "${SCENARIO_CLEAN}"
read -r -a METHOD_LIST <<< "${METHOD_CLEAN}"
if [[ ${#SCENARIO_LIST[@]} -eq 0 ]]; then
  SCENARIO_LIST=("all")
fi
if [[ ${#METHOD_LIST[@]} -eq 0 ]]; then
  METHOD_LIST=("baseline")
fi

for METHOD_ITEM in "${METHOD_LIST[@]}"; do
  for SCENARIO_ITEM in "${SCENARIO_LIST[@]}"; do
    for RUN_ID in $(seq ${RUN_START} ${RUN_END}); do
      RNG_SEED=$(( RANDOM_STATE_BASE + RUN_ID ))
      echo "[INFO] Run ${RUN_ID} seed=${RNG_SEED} scenario=${SCENARIO_ITEM} method=${METHOD_ITEM}"
      dai train \
        --train_csv "${TRAIN_CSV}" \
        --test_csv "${TEST_CSV}" \
        --external_csv "${EXTERNAL_CSV}" \
        --save_root "${SAVE_ROOT}" \
        --scenario "${SCENARIO_ITEM}" \
        --method "${METHOD_ITEM}" \
        --model_name "${MODEL}" \
        --scale_mode "${SCALE_MODE}" \
        --cv_splits "${CV_SPLITS}" \
        --early_stopping "${EARLY_STOPPING}" \
        --threshold 0.5 \
        --test_size "${TEST_SIZE}" \
        --n_repeats "${N_REPEATS}" \
        --random_state "${RNG_SEED}" \
        --run_range "${RUN_ID}-${RUN_ID}" \
        --total_steps "${TOTAL_STEPS}" \
        --kmm_weights_dir "${KMM_WEIGHTS_DIR}" \
        --cr_alpha_default "${CR_ALPHA_DEFAULT}" \
        $( [[ "${METHOD_ITEM}" == "ssda" || "${METHOD_ITEM}" == "regalign" ]] && echo --loss_mode "${LOSS_MODE}" ) \
        $( [[ -n "${MERGED_CSV}" ]] && echo --merged_csv "${MERGED_CSV}" ) \
        $( [[ "${REFIT}" == "1" ]] && echo --refit )
    done
  done
done
