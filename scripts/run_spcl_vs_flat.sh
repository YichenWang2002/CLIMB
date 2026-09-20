#!/bin/bash
# SPCL (new curriculum method) vs plain flat SFT, seed 42, matched budget.
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl
CUR=outputs/spcl_seed42

echo "=== [1/5] build SPCL curriculum (score + buckets + pacing) ==="
/root/miniconda3/bin/python -u -m curriculum.build_spcl \
  --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window 1,2,4 \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42

echo "=== [2/5] plain flat SFT, 3 epochs, seed 42 ==="
/root/miniconda3/bin/python -u -m training.sft_lora --mode flat \
  --stages "$TRAIN" --val "$VAL" \
  --run-name spcl_cmp_flat --epochs 3 --lr 1e-4 --seed 42

echo "=== [3/5] SPCL method: 3 paced rounds (fresh optimizer each round) ==="
/root/miniconda3/bin/python -u -m training.sft_lora --mode staged \
  --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
  --val "$VAL" --run-name spcl_cmp_method --epochs 1 --lr 1e-4 --seed 42

echo "=== [4/5] held-out test eval: flat ==="
/root/miniconda3/bin/python -u -m eval.evaluate --data "$TEST" \
  --adapter outputs/checkpoints/spcl_cmp_flat/stage1 \
  --out outputs/spcl_cmp_flat_test.json --batch-size 64 --save-generations

echo "=== [5/5] held-out test eval: SPCL method ==="
/root/miniconda3/bin/python -u -m eval.evaluate --data "$TEST" \
  --adapter outputs/checkpoints/spcl_cmp_method/stage3 \
  --out outputs/spcl_cmp_method_test.json --batch-size 64 --save-generations

echo "=== ALL DONE ==="
