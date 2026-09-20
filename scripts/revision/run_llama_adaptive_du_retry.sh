#!/usr/bin/env bash
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
BASE=models/llama32-1b
SIDECAR=outputs/revision/attribution_v2/seed42/curriculum/spcl/spcl_scores.jsonl
ROOT=outputs/revision/llama32_1b_adaptive_du_v2/seed42
CUR=$ROOT/curriculum/adaptive; CKPT=$ROOT/checkpoints; EVAL=$ROOT/eval
mkdir -p "$CUR" "$CKPT" "$EVAL"
# Reuse only the completed, pre-dose round-1 adapter from the failed D*R run.
INIT=./outputs/revision/llama32_1b_adaptive_dose_b8/seed42/checkpoints/adaptive_r1/stage1
for R in 2 3; do
  IDX=$((R-1))
  $PY -u -m curriculum.build_spcl --train data/train_aug10.jsonl --val data/val.jsonl \
    --out-dir "$CUR" --rounds 3 --buckets 4 --window 2,3,4 --scores-cache "$SIDECAR" \
    --only-round "$IDX" --base "$BASE" --seed 42 --adaptive-dose-du
  if [[ "$R" == 2 ]]; then PREV="$INIT"; else PREV="$CKPT/du_r2/stage1"; fi
  $PY -u -m training.sft_lora --mode flat --stages "$CUR/round${R}.jsonl" --val data/val.jsonl \
    --run-name "du_r${R}" --checkpoint-root "$CKPT" --base "$BASE" --epochs 1 --lr 1e-4 \
    --batch 8 --accum 8 --max-len 2048 --loss-type chunked_nll --seed $((42+R-1)) --init-adapter "$PREV"
done
FINAL=$CKPT/du_r3/stage1
$PY -u -m experiments.bt_ducl.strict_eval --data data/test.jsonl --adapter "$FINAL" --base "$BASE" \
  --out "$EVAL/spcl_adaptive_du_test_b64.json" --batch-size 64 --max-new 1400
$PY -u -m eval.eval_constrained --data data/test.jsonl --adapter "$FINAL" --base "$BASE" \
  --level topology --out "$EVAL/spcl_adaptive_du_scd_test_b64.json" --batch-size 64 --max-new 1400
echo "DU adaptive retry complete: $ROOT"
