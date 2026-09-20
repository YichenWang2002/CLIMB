#!/bin/bash
set -uo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl

for SEED in 43 44; do
  echo "=== flat seed ${SEED}: 3 epochs ==="
  $PY -u -m training.sft_lora --mode flat \
    --stages "$TRAIN" --val "$VAL" \
    --run-name flat_s${SEED} --epochs 3 --lr 1e-4 --seed ${SEED} || exit 1
  echo "=== flat seed ${SEED}: held-out test eval ==="
  $PY -u -m eval.evaluate --data "$TEST" \
    --adapter outputs/checkpoints/flat_s${SEED}/stage1 \
    --out "results/outputs/flat_s${SEED}_test.json" --batch-size 64 --save-generations || exit 1
  echo "=== flat seed ${SEED} DONE ==="
done
echo "=== FLAT MULTISEED DONE ==="
