#!/bin/bash
# Execution-based residual probes (SPCL-lambda gating signal, v2).
# R_b = fraction of stratified training tasks (100/bucket, seed 2026) the
# checkpoint still FAILS under greedy generation + symbolic execution.
# Training data only. Runs after the NLL probe batch frees the GPU.
set -uo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TRAIN=data/train_aug10.jsonl
OUT=outputs/revision/residual_probe
LLAMA=models/llama32-1b
QWEN=../model/qwen25-15b
LLAMA_SCORES=outputs/revision/attribution_v2/seed42/curriculum/spcl/spcl_scores.jsonl
LLAMA_CKPT=outputs/checkpoints/spcl_cmp_method_v2
QWEN_RUN=outputs/revision/qwen25_1p5b_chunked_b16_rerun/seed42
QWEN_SCORES=$QWEN_RUN/curriculum/spcl/spcl_scores.jsonl

execprobe () {  # label base adapter scores out
  if [[ -s "$5" ]]; then echo "=== exec probe $1: exists, skip ==="; return 0; fi
  echo "=== exec probe $1 ==="; date
  $PY -u -m curriculum.residual_probe --mode exec \
    --train "$TRAIN" --scores "$4" --base "$2" ${3:+--adapter "$3"} \
    --label "$1" --batch-size 64 --per-bucket 100 --sample-seed 2026 \
    --out "$5" || exit 1
}

mkdir -p "$OUT"

execprobe exec_llama_base "$LLAMA" "" "$LLAMA_SCORES" "$OUT/exec_llama_base.json"
execprobe exec_llama_s1   "$LLAMA" "$LLAMA_CKPT/stage1" "$LLAMA_SCORES" "$OUT/exec_llama_s1.json"
execprobe exec_llama_s2   "$LLAMA" "$LLAMA_CKPT/stage2" "$LLAMA_SCORES" "$OUT/exec_llama_s2.json"
execprobe exec_llama_s3   "$LLAMA" "$LLAMA_CKPT/stage3" "$LLAMA_SCORES" "$OUT/exec_llama_s3.json"

execprobe exec_qwen_base  "$QWEN" "" "$QWEN_SCORES" "$OUT/exec_qwen_base.json"
execprobe exec_qwen_s1    "$QWEN" "$QWEN_RUN/checkpoints/spcl/stage1" "$QWEN_SCORES" "$OUT/exec_qwen_s1.json"
execprobe exec_qwen_s2    "$QWEN" "$QWEN_RUN/checkpoints/spcl/stage2" "$QWEN_SCORES" "$OUT/exec_qwen_s2.json"
execprobe exec_qwen_s3    "$QWEN" "$QWEN_RUN/checkpoints/spcl/stage3" "$QWEN_SCORES" "$OUT/exec_qwen_s3.json"

echo "=== ALL EXEC PROBES DONE ==="; date
