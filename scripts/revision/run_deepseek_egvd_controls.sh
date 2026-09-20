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
ROOT=outputs/revision/deepseek_r1_qwen15b_matched_v2_b16/seed42
EVAL=$ROOT/eval
CKPT=$ROOT/checkpoints
mkdir -p "$EVAL"

run_one() {
  local name=$1
  local adapter=$2
  local constrained=$3
  local out="$EVAL/egvd_${name}_k8_t07.json"
  local args=(--data "$TEST" --adapter "$adapter" --base "$BASE"
    --out "$out" --k 8 --temperature 0.7 --top-p 0.95
    --batch-size 16 --max-new 1400 --seed 42)
  if [[ "$constrained" == "yes" ]]; then
    args+=(--constrained)
  fi
  echo "[$(date -Is)] starting $name"
  "$PY" -u -m eval.egvd "${args[@]}"
  echo "[$(date -Is)] finished $name"
}

run_one flat "$CKPT/deepseek_matched_flat/stage1" no
run_one spcl "$CKPT/deepseek_matched_spcl/stage3" no
run_one flat_scd "$CKPT/deepseek_matched_flat/stage1" yes

echo "[$(date -Is)] all DeepSeek EGVD control runs complete"
