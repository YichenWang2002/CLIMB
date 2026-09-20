#!/bin/bash
# Residual learning-distance probe (SPCL-λ Step 0).
# Measurement only: training data + existing checkpoints; no test/val access.
set -uo pipefail
DEVICE=${DEVICE:-cuda}
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

probe () {  # label base adapter scores out
  if [[ -s "$5" ]]; then echo "=== probe $1: exists, skip ==="; return 0; fi
  echo "=== probe $1 ==="; date
  $PY -u -m curriculum.residual_probe --mode score \
    --train "$TRAIN" --scores "$4" --base "$2" ${3:+--adapter "$3"} \
    --label "$1" --device "$DEVICE" --batch-size 8 --max-len 2560 --out "$5" || exit 1
}

mkdir -p "$OUT"

# Llama-3.2-1B v2 seed42 trajectory
probe llama_base "$LLAMA" "" "$LLAMA_SCORES" "$OUT/llama_base.json"
probe llama_s1   "$LLAMA" "$LLAMA_CKPT/stage1" "$LLAMA_SCORES" "$OUT/llama_s1.json"
probe llama_s2   "$LLAMA" "$LLAMA_CKPT/stage2" "$LLAMA_SCORES" "$OUT/llama_s2.json"
probe llama_s3   "$LLAMA" "$LLAMA_CKPT/stage3" "$LLAMA_SCORES" "$OUT/llama_s3.json"

# Qwen2.5-1.5B (chunked_b16_rerun seed42) trajectory
probe qwen_base  "$QWEN" "" "$QWEN_SCORES" "$OUT/qwen_base.json"
probe qwen_s1    "$QWEN" "$QWEN_RUN/checkpoints/spcl/stage1" "$QWEN_SCORES" "$OUT/qwen_s1.json"
probe qwen_s2    "$QWEN" "$QWEN_RUN/checkpoints/spcl/stage2" "$QWEN_SCORES" "$OUT/qwen_s2.json"
probe qwen_s3    "$QWEN" "$QWEN_RUN/checkpoints/spcl/stage3" "$QWEN_SCORES" "$OUT/qwen_s3.json"

echo "=== summarize ==="
$PY -u -m curriculum.residual_probe --mode summarize \
  --base-report "$OUT/llama_base.json" \
  --reports "$OUT/llama_s1.json,$OUT/llama_s2.json,$OUT/llama_s3.json" \
  --out "$OUT/summary_llama.json"
$PY -u -m curriculum.residual_probe --mode summarize \
  --base-report "$OUT/qwen_base.json" \
  --reports "$OUT/qwen_s1.json,$OUT/qwen_s2.json,$OUT/qwen_s3.json" \
  --out "$OUT/summary_qwen.json"
echo "=== ALL PROBES DONE ==="; date
