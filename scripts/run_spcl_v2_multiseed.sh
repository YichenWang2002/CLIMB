#!/bin/bash
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python

TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl

for SEED in 43 44; do
  CUR=outputs/spcl_v2_seed${SEED}
  echo "=== seed ${SEED}: build curriculum ==="
  $PY -u -m curriculum.build_spcl \
    --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
    --rounds 3 --buckets 4 --window 2,3,4 \
    --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
    --batch-size 8 --max-len 2560 --embed-device cpu --seed ${SEED}

  echo "=== seed ${SEED}: 3 paced rounds ==="
  $PY -u -m training.sft_lora --mode staged \
    --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
    --val "$VAL" --run-name spcl_v2_s${SEED} --epochs 1 --lr 1e-4 --seed ${SEED}

  echo "=== seed ${SEED}: held-out test eval ==="
  $PY -u -m eval.evaluate --data "$TEST" \
    --adapter outputs/checkpoints/spcl_v2_s${SEED}/stage3 \
    --out outputs/spcl_v2_s${SEED}_test.json --batch-size 64 --save-generations
done
echo "=== MULTISEED DONE ==="
