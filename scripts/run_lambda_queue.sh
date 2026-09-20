#!/usr/bin/env bash
# Sequential GPU queue: runs everything back-to-back so the GPU never idles.
# Order: λ s43 -> flat s43 -> λ s44 -> flat s44 -> Llama λ s42 -> Gemma (curriculum, flat, λ).
set -uo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
QLOG=outputs/revision/queue_run.log

echo "=== queue started $(date) ===" | tee -a "$QLOG"

# 0) wait for the in-flight Qwen lambda seed42 run to finish
while ! grep -q 'seed 42 DONE' outputs/revision/qwen25_1p5b_lambda_run_s42.log 2>/dev/null; do
  sleep 120
done
echo "=== in-flight s42 finished $(date) ===" | tee -a "$QLOG"

# 1) Qwen lambda seeds 43 / 44 interleaved with flat controls
echo "=== [1/6] Qwen lambda s43 ===" | tee -a "$QLOG"; date
SEEDS=43 EVAL_BATCH=8 DRY_RUN=0 bash scripts/revision/run_spcl_lambda.sh \
  >> outputs/revision/qwen25_1p5b_lambda_run_s43.log 2>&1
echo "=== [2/6] Qwen flat s43 ===" | tee -a "$QLOG"; date
SEEDS=43 EVAL_BATCH=8 bash scripts/revision/run_flat_backfill.sh \
  >> outputs/revision/qwen_flat_s43.log 2>&1
echo "=== [3/6] Qwen lambda s44 ===" | tee -a "$QLOG"; date
SEEDS=44 EVAL_BATCH=8 DRY_RUN=0 bash scripts/revision/run_spcl_lambda.sh \
  >> outputs/revision/qwen25_1p5b_lambda_run_s44.log 2>&1
echo "=== [4/6] Qwen flat s44 ===" | tee -a "$QLOG"; date
SEEDS=44 EVAL_BATCH=8 bash scripts/revision/run_flat_backfill.sh \
  >> outputs/revision/qwen_flat_s44.log 2>&1

# 2) Llama lambda s42 (degeneracy check, criterion B)
echo "=== [5/6] Llama lambda s42 ===" | tee -a "$QLOG"; date
BASE=models/llama32-1b \
BACKBONE_NAME=llama32_1b_lambda \
SIDECAR=outputs/revision/attribution_v2/seed42/curriculum/spcl/spcl_scores.jsonl \
SEEDS=42 EVAL_BATCH=8 BATCH=8 ACCUM=2 MAX_LEN=2560 LOSS_TYPE=nll DRY_RUN=0 \
  bash scripts/revision/run_spcl_lambda.sh \
  >> outputs/revision/llama32_1b_lambda_run_s42.log 2>&1

# 3) Gemma-2-2B: build curriculum (scoring on gemma base), flat, then lambda
echo "=== [6/6] Gemma curriculum + flat + lambda ===" | tee -a "$QLOG"; date
GEMMA=../model/gemma2-2b
GROOT=outputs/revision/gemma2_2b/seed42
if [[ ! -f "$GROOT/curriculum/spcl/spcl_scores.jsonl" ]]; then
  "$PY" -u -m curriculum.build_spcl \
    --train data/train_aug10.jsonl --val data/val.jsonl \
    --out-dir "$GROOT/curriculum/spcl" --rounds 3 --buckets 4 --window 2,3,4 \
    --boosts '1,1|1,1,1.5|0.5,0.75,1.5,3' --batch-size 8 --max-len 2048 \
    --embed-device cpu --base "$GEMMA" --seed 42 \
    >> outputs/revision/gemma_curriculum_build.log 2>&1
fi
BASE="$GEMMA" BACKBONE_NAME=gemma2_2b SEEDS=42 EVAL_BATCH=8 \
BATCH=8 ACCUM=2 MAX_LEN=2048 LOSS_TYPE=nll \
  bash scripts/revision/run_flat_backfill.sh \
  >> outputs/revision/gemma_flat_s42.log 2>&1
BASE="$GEMMA" BACKBONE_NAME=gemma2_2b_lambda \
SIDECAR="$GROOT/curriculum/spcl/spcl_scores.jsonl" \
SEEDS=42 EVAL_BATCH=8 BATCH=8 ACCUM=2 MAX_LEN=2048 LOSS_TYPE=nll DRY_RUN=0 \
  bash scripts/revision/run_spcl_lambda.sh \
  >> outputs/revision/gemma2_2b_lambda_run_s42.log 2>&1

echo "=== ALL QUEUE DONE $(date) ===" | tee -a "$QLOG"
