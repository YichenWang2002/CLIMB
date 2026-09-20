#!/bin/bash
set -uo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl

for SEED in 45 46; do
  echo "=== flat seed ${SEED} ==="
  $PY -u -m training.sft_lora --mode flat \
    --stages "$TRAIN" --val "$VAL" \
    --run-name flat_s${SEED} --epochs 3 --lr 1e-4 --seed ${SEED} || exit 1
  $PY -u -m eval.evaluate --data "$TEST" \
    --adapter outputs/checkpoints/flat_s${SEED}/stage1 \
    --out "results/outputs/flat_s${SEED}_test.json" --batch-size 64 --save-generations || exit 1
  echo "=== v2 seed ${SEED}: build ==="
  CUR=outputs/spcl_v2_seed${SEED}
  $PY -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
    --rounds 3 --buckets 4 --window 2,3,4 \
    --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
    --batch-size 8 --max-len 2560 --embed-device cpu --seed ${SEED} || exit 1
  echo "=== v2 seed ${SEED}: train ==="
  $PY -u -m training.sft_lora --mode staged \
    --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
    --val "$VAL" --run-name spcl_v2_s${SEED} --epochs 1 --lr 1e-4 --seed ${SEED} || exit 1
  $PY -u -m eval.evaluate --data "$TEST" \
    --adapter outputs/checkpoints/spcl_v2_s${SEED}/stage3 \
    --out "results/outputs/spcl_v2_s${SEED}_test.json" --batch-size 64 --save-generations || exit 1
  echo "=== seed ${SEED} DONE ==="
done
echo "=== SEEDS 45/46 DONE ==="
