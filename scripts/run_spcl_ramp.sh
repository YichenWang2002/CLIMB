#!/bin/bash
# SPCL boost-shape robustness: zero-parameter monotone emphasis ramps.
# Identical protocol to run_spcl_v2.sh (same train/val/test, window 2,3,4,
# seed 42, 3 rounds x 1 epoch, lr 1e-4); only the boost matrix changes.
#
# Usage: run_spcl_ramp.sh <variant> <boosts>
#   linear  "1,1|1,2,3|1,2,3,4"              (B3:B0 ratio 4, zero free params)
#   geo15   "1,1|1,1.5,2.25|1,1.5,2.25,3.375" (mild, ratio ~3.4)
#   geo2    "1,1|1,2,4|1,2,4,8"              (aggressive, ratio 8)
# Reference: manual v2 boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" (ratio 6) -> 62.0%.
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python

VARIANT=${1:?variant name}
BOOSTS=${2:?boosts string}

TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl
CUR=outputs/spcl_ramp_${VARIANT}_seed42

echo "=== [1/3] build SPCL ramp curriculum ($VARIANT, boosts=$BOOSTS) ==="
$PY -u -m curriculum.build_spcl \
  --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window 2,3,4 \
  --boosts "$BOOSTS" \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42

echo "=== [2/3] SPCL ramp $VARIANT: 3 paced rounds ==="
$PY -u -m training.sft_lora --mode staged \
  --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
  --val "$VAL" --run-name "spcl_ramp_$VARIANT" --epochs 1 --lr 1e-4 --seed 42

echo "=== [3/3] held-out test eval: SPCL ramp $VARIANT ==="
$PY -u -m eval.evaluate --data "$TEST" \
  --adapter "outputs/checkpoints/spcl_ramp_$VARIANT/stage3" \
  --out "outputs/spcl_ramp_${VARIANT}_test.json" --batch-size 64 --save-generations

echo "=== RAMP $VARIANT DONE ==="
