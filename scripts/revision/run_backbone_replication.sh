#!/usr/bin/env bash
# Qwen/Llama backbone replication: flat, SPCL, strict evaluation, and SCD.
#
# Default BASE is the local Qwen2.5-1.5B-Instruct snapshot. The protocol keeps
# train/val/test fixed, uses equal three-epoch optimizer budgets, and isolates
# every seed under outputs/revision/<backbone>/seed<seed>.
#
# Dry run (no model loading): DRY_RUN=1 bash pipeline/scripts/revision/run_backbone_replication.sh
# Real run (one seed first): SEEDS=42 DRY_RUN=0 bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/root/miniconda3/bin/python}"
BASE="${BASE:-../model/qwen25-15b}"
BACKBONE_NAME="${BACKBONE_NAME:-qwen25_1p5b}"
SEEDS="${SEEDS:-42 43 44}"
ROOT="${ROOT:-outputs/revision/${BACKBONE_NAME}}"
TRAIN="${TRAIN:-data/train_aug10.jsonl}"
VAL="${VAL:-data/val.jsonl}"
TEST="${TEST:-data/test.jsonl}"
DRY_RUN="${DRY_RUN:-1}"
RUN_SCD="${RUN_SCD:-1}"
# Historical Llama runs used Trainer's RandomSampler.  SequentialSampler is
# retained only as an explicit order-ablation setting (PRESERVE_ORDER=1).
PRESERVE_ORDER="${PRESERVE_ORDER:-0}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-2}"
CURRICULUM_BATCH="${CURRICULUM_BATCH:-$BATCH}"
MAX_LEN="${MAX_LEN:-2048}"
EVAL_BATCH="${EVAL_BATCH:-4}"
LOSS_TYPE="${LOSS_TYPE:-chunked_nll}"
CURRICULUM_SOURCE="${CURRICULUM_SOURCE:-}"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

assert_inputs() {
  [[ -f "$TRAIN" && -f "$VAL" && -f "$TEST" ]] || {
    echo "missing train/val/test input" >&2; exit 2;
  }
  [[ -d "$BASE" ]] || {
    echo "BASE is not a local model directory: $BASE" >&2; exit 2;
  }
}

for SEED in $SEEDS; do
  SEED_ROOT="$ROOT/seed${SEED}"
  CUR="$SEED_ROOT/curriculum/spcl"
  CKPT="$SEED_ROOT/checkpoints"
  EVAL="$SEED_ROOT/eval"
  MANIFEST="$SEED_ROOT/manifest.json"
  if [[ -e "$MANIFEST" && "$DRY_RUN" != 1 && "${OVERWRITE:-0}" != 1 ]]; then
    echo "refusing to reuse existing run manifest: $MANIFEST" >&2
    echo "Choose a new ROOT or remove only this explicitly generated run." >&2
    exit 3
  fi
  mkdir -p "$SEED_ROOT"
  run "$PY" -u -m experiments.revision.protocol manifest \
    --phase "backbone_replication/${BACKBONE_NAME}" --status planned \
    --base "$BASE" --seed "$SEED" --train "$TRAIN" --val "$VAL" --test "$TEST" \
    --optimizer-steps "{\"flat_epochs\":3,\"spcl_rounds\":3,\"effective_batch\":$((BATCH * ACCUM)),\"micro_batch\":$BATCH,\"gradient_accumulation\":$ACCUM,\"curriculum_score_batch\":$CURRICULUM_BATCH,\"loss_type\":\"$LOSS_TYPE\",\"max_len\":$MAX_LEN,\"staged_sampler\":\"$([[ \"$PRESERVE_ORDER\" == 1 ]] && echo sequential || echo random)\"}" \
    --decode-config "{\"greedy\":true,\"max_new_tokens\":1400,\"eval_batch\":$EVAL_BATCH}" --out "$MANIFEST" \
    --overwrite

  if [[ -n "$CURRICULUM_SOURCE" ]]; then
    [[ -f "$CURRICULUM_SOURCE/report.json" ]] || {
      echo "CURRICULUM_SOURCE lacks report.json: $CURRICULUM_SOURCE" >&2; exit 2;
    }
    mkdir -p "$CUR"
    run cp -a "$CURRICULUM_SOURCE/." "$CUR/"
  else
    run "$PY" -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" \
      --out-dir "$CUR" --rounds 3 --buckets 4 --window 2,3,4 \
      --boosts '1,1|1,1,1.5|0.5,0.75,1.5,3' --batch-size "$CURRICULUM_BATCH" --max-len "$MAX_LEN" \
      --embed-device cpu --base "$BASE" --seed "$SEED"
  fi

  run "$PY" -u -m training.sft_lora --mode flat --stages "$TRAIN" --val "$VAL" \
    --run-name flat --checkpoint-root "$CKPT" --base "$BASE" \
    --epochs 3 --lr 1e-4 --batch "$BATCH" --accum "$ACCUM" --max-len "$MAX_LEN" \
    --loss-type "$LOSS_TYPE" --seed "$SEED"
  STAGED_ARGS=("$PY" -u -m training.sft_lora --mode staged \
    --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
    --val "$VAL" --run-name spcl --checkpoint-root "$CKPT" --base "$BASE" \
    --epochs 1 --lr 1e-4 --batch "$BATCH" --accum "$ACCUM" --max-len "$MAX_LEN" \
    --loss-type "$LOSS_TYPE" --seed "$SEED")
  if [[ "$PRESERVE_ORDER" == 1 ]]; then
    STAGED_ARGS+=(--preserve-order)
  fi
  run "${STAGED_ARGS[@]}"

  run "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$CKPT/flat/stage1" --base "$BASE" --out "$EVAL/flat_test.json" \
    --batch-size "$EVAL_BATCH" --max-new 1400
  run "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$CKPT/spcl/stage3" --base "$BASE" --out "$EVAL/spcl_test.json" \
    --batch-size "$EVAL_BATCH" --max-new 1400
  if [[ "$RUN_SCD" == 1 ]]; then
    run "$PY" -u -m eval.eval_constrained --data "$TEST" \
      --adapter "$CKPT/flat/stage1" --base "$BASE" --level topology \
      --out "$EVAL/flat_scd_test.json" --batch-size "$EVAL_BATCH" --max-new 1400
    run "$PY" -u -m eval.eval_constrained --data "$TEST" \
      --adapter "$CKPT/spcl/stage3" --base "$BASE" --level topology \
      --out "$EVAL/spcl_scd_test.json" --batch-size "$EVAL_BATCH" --max-new 1400
  fi
done

if [[ "$DRY_RUN" != 1 ]]; then
  run "$PY" -u -m experiments.revision.aggregate_revision --root "$ROOT" \
    --baseline flat --out "$ROOT/aggregate.json" --bootstrap-samples 10000
  "$PY" -u -m experiments.revision.protocol manifest \
    --phase "backbone_replication/${BACKBONE_NAME}" --status complete \
    --base "$BASE" --train "$TRAIN" --val "$VAL" --test "$TEST" \
    --optimizer-steps "{\"flat_epochs\":3,\"spcl_rounds\":3,\"effective_batch\":$((BATCH * ACCUM)),\"micro_batch\":$BATCH,\"gradient_accumulation\":$ACCUM,\"curriculum_score_batch\":$CURRICULUM_BATCH,\"loss_type\":\"$LOSS_TYPE\",\"max_len\":$MAX_LEN,\"staged_sampler\":\"$([[ \"$PRESERVE_ORDER\" == 1 ]] && echo sequential || echo random)\"}" \
    --decode-config "{\"greedy\":true,\"max_new_tokens\":1400,\"eval_batch\":$EVAL_BATCH}" \
    --out "$ROOT/completed_manifest.json" --overwrite
fi
echo "Backbone replication protocol emitted. DRY_RUN=$DRY_RUN ROOT=$ROOT"
