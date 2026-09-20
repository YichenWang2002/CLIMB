#!/bin/bash
# SPCL (Structure-Paced Curriculum Learning): score difficulty/utility once,
# build R=3 rounds over K=4 buckets with expanding windows (2,3,4) and the
# frozen dose table, then run three 1-epoch stages (fresh optimizer/LR per
# round, only the LoRA adapter carries over). Budget == flat 3 epochs.
# Usage: bash scripts/run_spcl.sh [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${1:-42}"
CUR="outputs/spcl_s${SEED}"

echo "=== [1/3] SPCL curriculum (two-view difficulty + utility, K=4, windows 2,3,4) ==="
python -u -m curriculum.build_spcl \
  --train data/train_aug10.jsonl --val data/val.jsonl --out-dir "${CUR}" \
  --rounds 3 --buckets 4 --window 2,3,4 \
  --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed "${SEED}"

echo "=== [2/3] staged training: 3 x 1 epoch ==="
python -u -m training.sft_lora --mode staged \
  --stages "${CUR}/round1.jsonl" "${CUR}/round2.jsonl" "${CUR}/round3.jsonl" \
  --val data/val.jsonl \
  --run-name "spcl_s${SEED}" --epochs 1 --lr 1e-4 --seed "${SEED}"

echo "=== [3/3] held-out test eval ==="
python -u -m eval.evaluate --data data/test.jsonl \
  --adapter "outputs/checkpoints/spcl_s${SEED}/stage3" \
  --out "results/spcl_s${SEED}.json" --batch-size 64 --save-generations
