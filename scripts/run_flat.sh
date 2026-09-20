#!/bin/bash
# Flat SFT baseline (BTBase): 3 epochs, shuffled sampling, LoRA r16/a32.
# Usage: bash scripts/run_flat.sh [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${1:-42}"

python -u -m training.sft_lora --mode flat \
  --stages data/train_aug10.jsonl \
  --val data/val.jsonl \
  --run-name "btgenbot_ma_s${SEED}" --epochs 3 --lr 1e-4 --seed "${SEED}"

python -u -m eval.evaluate --data data/test.jsonl \
  --adapter "outputs/checkpoints/btgenbot_ma_s${SEED}/stage1" \
  --out "results/btgenbot_ma_s${SEED}.json" --batch-size 64 --save-generations
