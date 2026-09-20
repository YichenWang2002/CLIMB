#!/usr/bin/env bash
set -euo pipefail

cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

PY=/root/miniconda3/bin/python
BASE=../model/DeepSeek-R1-Distill-Qwen-1.5B
TEST=data/test.jsonl
ROOT_OUT=outputs/revision/deepseek_r1_qwen15b_matched_v2_b16/seed42
CKPT=$ROOT_OUT/checkpoints
EVAL=$ROOT_OUT/eval
mkdir -p "$EVAL"

echo "[1/4] flat strict, batch 64"
if [ ! -s "$EVAL/flat_test.json" ]; then $PY -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
  --adapter "$CKPT/deepseek_matched_flat/stage1" --base "$BASE" \
  --out "$EVAL/flat_test.json" --batch-size 64 --max-new 1400; fi

echo "[2/4] SPCL strict, batch 64"
if [ ! -s "$EVAL/spcl_test.json" ]; then $PY -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
  --adapter "$CKPT/deepseek_matched_spcl/stage3" --base "$BASE" \
  --out "$EVAL/spcl_test.json" --batch-size 64 --max-new 1400; fi

echo "[3/4] flat + SCD topology, batch 64"
if [ ! -s "$EVAL/flat_scd_test.json" ]; then $PY -u -m eval.eval_constrained --data "$TEST" \
  --adapter "$CKPT/deepseek_matched_flat/stage1" --base "$BASE" \
  --out "$EVAL/flat_scd_test.json" --batch-size 64 --max-new 1400 --level topology; fi

echo "[4/4] SPCL + SCD topology, batch 64"
if [ ! -s "$EVAL/spcl_scd_test.json" ]; then $PY -u -m eval.eval_constrained --data "$TEST" \
  --adapter "$CKPT/deepseek_matched_spcl/stage3" --base "$BASE" \
  --out "$EVAL/spcl_scd_test.json" --batch-size 64 --max-new 1400 --level topology; fi

echo "DONE: batch-64 evaluations written to $EVAL"
