#!/bin/bash
# Flat supervised fine-tuning baseline (BTGenBot-MA style).
# Learning rate / LoRA configuration are intentionally left to the user.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLIMB_LR:?export CLIMB_LR (learning rate; withheld in this release)}"
: "${CLIMB_LORA_R:?export CLIMB_LORA_R}"
: "${CLIMB_LORA_ALPHA:?export CLIMB_LORA_ALPHA}"

python -u -m training.sft_lora --mode flat \
  --stages data/train.jsonl \
  --val data/val.jsonl \
  --run-name "btbase_s${SEED:-42}" --epochs 3 --lr "$CLIMB_LR" \
  --lora-r "$CLIMB_LORA_R" --lora-alpha "$CLIMB_LORA_ALPHA" \
  --seed "${SEED:-42}"

python -u -m eval.evaluate --data data/test.jsonl \
  --adapter "outputs/checkpoints/btbase_s${SEED:-42}/stage1" \
  --out "outputs/results/btbase_s${SEED:-42}.json" --batch-size 64 --save-generations
