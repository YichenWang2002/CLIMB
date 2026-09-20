#!/bin/bash
# Nine-epoch strong flat SFT reference for the ICASSP experiments.
# The historical run name is retained. Once ECFT's training budget is fixed,
# adjust this control to match its optimizer steps or target-token count.
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VAL=data/val.jsonl
TEST=data/test.jsonl
NAME=baseline_flat_cm

python3 -u -m training.sft_lora --mode flat \
  --stages data/train_aug10.jsonl \
  --val "$VAL" --run-name "$NAME" --epochs 9 --lr 1e-4

python3 -u -m eval.evaluate --data "$VAL" \
  --adapter "outputs/checkpoints/$NAME/stage1" \
  --out "${NAME}_val.json" --batch-size 64
python3 -u -m eval.evaluate --data "$TEST" \
  --adapter "outputs/checkpoints/$NAME/stage1" \
  --out "${NAME}_test.json" --batch-size 64
