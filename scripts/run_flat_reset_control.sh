#!/bin/bash
# Fair Flat SFT control: the same three 1-epoch invocations and optimizer/LR
# resets as MT-DUCL, with a fresh deterministic random permutation each round.
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl
ROOT=outputs/flat_reset_control
NAME=flat_reset_control
mkdir -p "$ROOT"

PREV=""
for round in 0 1 2; do
  ORDERED="$ROOT/random_round${round}.jsonl"
  if [ ! -s "$ORDERED" ]; then
    python3 -u -m curriculum.make_random_order \
      --input "$TRAIN" --output "$ORDERED" --seed $((42 + round))
  fi
  RUN_DIR="outputs/checkpoints/${NAME}_round$((round + 1))"
  echo "=== Flat reset round $((round + 1)): one epoch ==="
  INIT_ARG=()
  if [ -n "$PREV" ]; then
    INIT_ARG=(--init-adapter "$PREV")
  fi
  python3 -u -m training.sft_lora --mode flat \
    --stages "$ORDERED" --val "$VAL" \
    --run-name "${NAME}_round$((round + 1))" \
    --epochs 1 --lr 1e-4 --preserve-order --seed 42 "${INIT_ARG[@]}"
  PREV="$RUN_DIR/stage1"
done

echo "=== Flat reset validation ==="
python3 -u -m eval.evaluate --data "$VAL" \
  --adapter "$PREV" --out flat_reset_control_val.json \
  --batch-size 64 --save-generations

echo "=== Flat reset held-out test ==="
python3 -u -m eval.evaluate --data "$TEST" \
  --adapter "$PREV" --out flat_reset_control_test.json \
  --batch-size 64 --save-generations

echo "=== Flat reset control complete ==="
