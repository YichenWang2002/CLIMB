#!/usr/bin/env bash
# Flat backfill (paired controls): defaults to Qwen2.5-1.5B seeds 43/44.
# Protocol mirrors run_backbone_replication.sh flat arm; eval batch raised
# to 8 (greedy decode is batch-invariant, so results are unchanged).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/root/miniconda3/bin/python}"

BASE="${BASE:-../model/qwen25-15b}"
BACKBONE_NAME="${BACKBONE_NAME:-qwen25_1p5b}"
SEEDS="${SEEDS:-43 44}"
TRAIN="${TRAIN:-data/train_aug10.jsonl}"
VAL="${VAL:-data/val.jsonl}"
TEST="${TEST:-data/test.jsonl}"
BATCH="${BATCH:-16}"
ACCUM="${ACCUM:-1}"
MAX_LEN="${MAX_LEN:-2048}"
EVAL_BATCH="${EVAL_BATCH:-8}"
LOSS_TYPE="${LOSS_TYPE:-chunked_nll}"
RUN_SCD="${RUN_SCD:-1}"

for SEED in $SEEDS; do
  SEED_ROOT="outputs/revision/${BACKBONE_NAME}/seed${SEED}"
  CKPT="$SEED_ROOT/checkpoints"
  EVAL="$SEED_ROOT/eval"
  MANIFEST="$SEED_ROOT/manifest.json"
  if [[ -e "$MANIFEST" ]]; then
    echo "refusing to reuse existing manifest: $MANIFEST" >&2; exit 3
  fi
  mkdir -p "$SEED_ROOT"
  "$PY" -u -m experiments.revision.protocol manifest \
    --phase "backbone_replication_flatfill/${BACKBONE_NAME}" --status planned \
    --base "$BASE" --seed "$SEED" --train "$TRAIN" --val "$VAL" --test "$TEST" \
    --optimizer-steps "{\"flat_epochs\":3,\"spcl_rounds\":0,\"effective_batch\":$((BATCH * ACCUM)),\"micro_batch\":$BATCH,\"gradient_accumulation\":$ACCUM,\"loss_type\":\"$LOSS_TYPE\",\"max_len\":$MAX_LEN,\"staged_sampler\":\"random\"}" \
    --decode-config "{\"greedy\":true,\"max_new_tokens\":1400,\"eval_batch\":$EVAL_BATCH}" \
    --out "$MANIFEST" --overwrite

  echo "=== [${BACKBONE_NAME} flat s${SEED}] train ==="; date
  "$PY" -u -m training.sft_lora --mode flat --stages "$TRAIN" --val "$VAL" \
    --run-name flat --checkpoint-root "$CKPT" --base "$BASE" \
    --epochs 3 --lr 1e-4 --batch "$BATCH" --accum "$ACCUM" --max-len "$MAX_LEN" \
    --loss-type "$LOSS_TYPE" --seed "$SEED"

  echo "=== [${BACKBONE_NAME} flat s${SEED}] strict eval ==="; date
  "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$CKPT/flat/stage1" --base "$BASE" --out "$EVAL/flat_test.json" \
    --batch-size "$EVAL_BATCH" --max-new 1400
  if [[ "$RUN_SCD" == 1 ]]; then
    echo "=== [${BACKBONE_NAME} flat s${SEED}] SCD eval ==="; date
    "$PY" -u -m eval.eval_constrained --data "$TEST" \
      --adapter "$CKPT/flat/stage1" --base "$BASE" --level topology \
      --out "$EVAL/flat_scd_test.json" --batch-size "$EVAL_BATCH" --max-new 1400
  fi
  echo "=== ${BACKBONE_NAME} flat seed ${SEED} DONE ==="; date
done
echo "FLAT BACKFILL DONE"
