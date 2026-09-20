#!/bin/bash
# Complete the 1.5B ablation column: the two missing cells.
#   row CLIMB (full)  : SPCL on train_aug10 (6000) -> eval +-SCD
#   row w/o SPCL      : flat SFT on train_aug10 (3 epochs) -> eval +SCD
# (w/o supervision and w/o SCD rows already exist from prior runs.)
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3

BASE15=../model/DeepSeek-R1-Distill-Qwen-1.5B
TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl

echo "=== [1/3] CLIMB full on 1.5B: SPCL curriculum + staged training ==="
$PY -u -m curriculum.build_spcl \
  --train "$TRAIN" --val "$VAL" --out-dir outputs/spcl_full_ds15_seed42 --base "$BASE15" \
  --rounds 3 --buckets 4 --window 2,3,4 \
  --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42
$PY -u -m training.sft_lora --mode staged --base "$BASE15" \
  --stages outputs/spcl_full_ds15_seed42/round1.jsonl \
           outputs/spcl_full_ds15_seed42/round2.jsonl \
           outputs/spcl_full_ds15_seed42/round3.jsonl \
  --val "$VAL" --run-name spcl_full_ds15 --epochs 1 --lr 1e-4 --seed 42
$PY -u -m eval.evaluate --data "$TEST" --base "$BASE15" \
  --adapter outputs/checkpoints/spcl_full_ds15/stage3 \
  --out results/outputs/spcl_full_ds15_test.json --batch-size 64 --save-generations
$PY -u -m eval.eval_constrained --data "$TEST" --base "$BASE15" \
  --adapter outputs/checkpoints/spcl_full_ds15/stage3 \
  --out outputs/spcl_full_ds15_scd_test.json --batch-size 48 --level topology

echo "=== [2/3] w/o SPCL on 1.5B: flat SFT (matched 3-epoch recipe) ==="
$PY -u -m training.sft_lora --mode flat --base "$BASE15" \
  --stages "$TRAIN" --val "$VAL" --run-name flat_ds15 --epochs 3 --lr 1e-4 --seed 42
$PY -u -m eval.eval_constrained --data "$TEST" --base "$BASE15" \
  --adapter outputs/checkpoints/flat_ds15/stage1 \
  --out outputs/flat_ds15_scd_test.json --batch-size 48 --level topology

echo "=== [3/3] w/o SCD on 1.5B: free decoding from the full checkpoint ==="
$PY -u -m eval.evaluate --data "$TEST" --base "$BASE15" \
  --adapter outputs/checkpoints/spcl_full_ds15/stage3 \
  --out results/outputs/spcl_full_ds15_noscd.json --batch-size 64 --save-generations

echo "=== DS15 COLUMN COMPLETE ==="
