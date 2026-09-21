#!/bin/bash
# SPCL curriculum + staged training (reference pipeline).
# The curriculum hyperparameters (bucket count, expanding-window schedule,
# per-bucket dose multipliers, learning rate, LoRA configuration) are
# intentionally withheld -- provide your own via the environment variables.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLIMB_LR:?export CLIMB_LR (learning rate; withheld in this release)}"
: "${CLIMB_LORA_R:?export CLIMB_LORA_R}"
: "${CLIMB_LORA_ALPHA:?export CLIMB_LORA_ALPHA}"
: "${CLIMB_WINDOW:?export CLIMB_WINDOW (bucket-prefix sizes per round, e.g. a comma list)}"
: "${CLIMB_DOSE:?export CLIMB_DOSE (per-round bucket multipliers: rounds separated by '|')}"

SEED="${SEED:-42}"
CUR="outputs/spcl_s${SEED}"

echo "=== [1/3] SPCL curriculum (two-view difficulty + utility) ==="
python -u -m curriculum.build_spcl \
  --train data/train.jsonl --val data/val.jsonl --out-dir "${CUR}" \
  --rounds 3 --buckets 4 --window "$CLIMB_WINDOW" \
  --boosts "$CLIMB_DOSE" \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed "${SEED}"

echo "=== [2/3] staged training: 3 x 1 epoch ==="
python -u -m training.sft_lora --mode staged \
  --stages "${CUR}/round1.jsonl" "${CUR}/round2.jsonl" "${CUR}/round3.jsonl" \
  --val data/val.jsonl \
  --run-name "spcl_s${SEED}" --epochs 1 --lr "$CLIMB_LR" \
  --lora-r "$CLIMB_LORA_R" --lora-alpha "$CLIMB_LORA_ALPHA" \
  --seed "${SEED}"

echo "=== [3/3] held-out test eval ==="
python -u -m eval.evaluate --data data/test.jsonl \
  --adapter "outputs/checkpoints/spcl_s${SEED}/stage3" \
  --out "outputs/results/spcl_s${SEED}.json" --batch-size 64 --save-generations
