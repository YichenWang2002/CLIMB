#!/bin/bash
# Five pre-specified seeds (42-46), no selection: flat SFT and SPCL, with the
# greedy + SCD evals of the deployed stack. Reports aggregate later via
# eval/paired_test and seed-level paired t-tests.
# Usage: bash scripts/run_multiseed.sh [SEEDS...]   (default: 42 43 44 45 46)
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
  echo "=== seed ${SEED}: flat SFT ==="
  bash scripts/run_flat.sh "${SEED}" || { echo "flat s${SEED} FAILED"; exit 1; }
  echo "=== seed ${SEED}: SPCL ==="
  bash scripts/run_spcl.sh "${SEED}" || { echo "spcl s${SEED} FAILED"; exit 1; }
  echo "=== seed ${SEED}: deployed stack (SPCL adapter + SCD decoding) ==="
  bash scripts/run_scd.sh "outputs/checkpoints/spcl_s${SEED}/stage3" "spcl_s${SEED}_scd" || exit 1
done
echo "=== ALL SEEDS DONE ==="
for SEED in "${SEEDS[@]}"; do
  for R in btgenbot_ma_s${SEED} spcl_s${SEED} spcl_s${SEED}_scd; do
    python -c "import json;d=json.load(open('results/${R}.json'));print('${R}', round(100*d['exec_success_rate'],2))"
  done
done
