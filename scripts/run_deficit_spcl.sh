#!/bin/bash
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/miniconda3/bin/python -u -m curriculum.run_deficit_spcl \
  --train data/train_aug10.jsonl \
  --val data/val.jsonl \
  --test data/test.jsonl \
  --out-dir outputs/pd_spcl_seed42 \
  --run-name pd_spcl_s42 \
  --buckets 4 --window 2,3,4 --probe-per-bucket 100 \
  --score-batch-size 8 --train-batch-size 4 --accum 4 \
  --max-len 2560 --lr 1e-4 --seed 42
