#!/bin/bash
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
V2=outputs/checkpoints/spcl_cmp_method/stage3

echo "=== [1/3] RFT sampling + execution verification ==="
$PY -u -m curriculum.rft_sample_verify \
  --train data/train_aug10.jsonl --adapter "$V2" \
  --out-dir outputs/v5_rft_seed42 --n-tasks 1200 --k 6 \
  --temperature 0.8 --top-p 0.95 --max-new 1400 \
  --batch-size 8 --keep-per-task 3 --seed 42

echo "=== [2/3] RFT SFT (1 epoch, lr 5e-5, from v2 adapter) ==="
$PY -u -m training.sft_lora --mode flat \
  --stages outputs/v5_rft_seed42/rft_train.jsonl \
  --val data/val.jsonl \
  --run-name v5_rft --epochs 1 --lr 5e-5 --seed 42 \
  --init-adapter "$V2"

echo "=== [3/3] held-out test eval: v5 ==="
$PY -u -m eval.evaluate --data data/test.jsonl \
  --adapter outputs/checkpoints/v5_rft/stage1 \
  --out outputs/v5_rft_test.json --batch-size 64 --save-generations

echo "=== V5 DONE ==="
