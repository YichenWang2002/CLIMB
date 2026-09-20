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
CUR=outputs/spcl_v2_seed42

echo "=== [1/3] build SPCL v2 curriculum (window 2,3,4 + boosts) ==="
$PY -u -m curriculum.build_spcl \
  --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window 2,3,4 \
  --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42

echo "=== [2/3] SPCL v2: 3 paced rounds ==="
$PY -u -m training.sft_lora --mode staged \
  --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
  --val "$VAL" --run-name spcl_cmp_method_v2 --epochs 1 --lr 1e-4 --seed 42

echo "=== [3/3] held-out test eval: SPCL v2 ==="
$PY -u -m eval.evaluate --data "$TEST" \
  --adapter outputs/checkpoints/spcl_cmp_method_v2/stage3 \
  --out outputs/spcl_cmp_method_v2_test.json --batch-size 64 --save-generations

echo "=== V2 DONE ==="
