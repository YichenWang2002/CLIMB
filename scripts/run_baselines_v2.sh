#!/bin/bash
# Standard three-epoch flat SFT baseline retained for the ICASSP project.
# The historical run name `baseline_random_v2` is kept for compatibility with
# the existing checkpoint and result filenames.
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl
NAME=baseline_random_v2

python3 -u -m training.sft_lora --mode flat \
  --stages "$TRAIN" --val "$VAL" --run-name "$NAME" \
  --epochs 3 --lr 1e-4

python3 -u -m eval.evaluate --data "$VAL" \
  --adapter "outputs/checkpoints/$NAME/stage1" \
  --out "${NAME}_val.json" --batch-size 64
python3 -u -m eval.evaluate --data "$TEST" \
  --adapter "outputs/checkpoints/$NAME/stage1" \
  --out "${NAME}_test.json" --batch-size 64
