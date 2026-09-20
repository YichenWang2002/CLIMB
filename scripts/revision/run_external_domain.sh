#!/usr/bin/env bash
# Generate and evaluate the held-out kitchen domain. No kitchen row is added
# to train/validation; the split validator is a hard gate before inference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
PY="${PY:-/root/miniconda3/bin/python}"
BASE="${BASE:-../model/qwen25-15b}"
DATA="${DATA:-outputs/revision/external_domain/kitchen.jsonl}"
TRAIN="${TRAIN:-data/train_aug10.jsonl}"
VAL="${VAL:-data/val.jsonl}"
ADAPTER_ROOT="${ADAPTER_ROOT:-outputs/revision/qwen25_1p5b/seed42/checkpoints}"
DRY_RUN="${DRY_RUN:-1}"
N="${N:-300}"
NL_MODE="${NL_MODE:-template}"
EVAL_BATCH="${EVAL_BATCH:-4}"
RUN_SCD="${RUN_SCD:-0}"

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}

if [[ ! -f "$DATA" ]]; then
  BUILD_ARGS=("$PY" -u -m experiments.revision.build_external_domain --out "$DATA" \
    --train "$TRAIN" --val "$VAL" --n "$N" --nl-mode "$NL_MODE" \
    --workers 1 --nl-workers 8)
  if [[ "$DRY_RUN" == 1 ]]; then
    run "${BUILD_ARGS[@]}" --dry-run
  else
    run "${BUILD_ARGS[@]}"
  fi
fi
EVAL_ARGS=(--data "$DATA" --train "$TRAIN" --val "$VAL" --base "$BASE" --batch-size "$EVAL_BATCH")
if [[ "$DRY_RUN" == 1 ]]; then
  printf '+ %q ' "$PY" -u -m experiments.revision.evaluate_external_domain "${EVAL_ARGS[@]}"
  printf '%q ' --adapter "$ADAPTER_ROOT/flat/stage1" --out "${DATA%.jsonl}_flat.json"
  printf '%s\n' --dry-run
  printf '+ %q ' "$PY" -u -m experiments.revision.evaluate_external_domain "${EVAL_ARGS[@]}"
  printf '%q ' --adapter "$ADAPTER_ROOT/spcl/stage3" --out "${DATA%.jsonl}_spcl.json"
  printf '%s\n' --dry-run
  if [[ "$RUN_SCD" == 1 ]]; then
    printf '+ %q ' "$PY" -u -m eval.eval_constrained --data "$DATA"
    printf '%q ' --adapter "$ADAPTER_ROOT/flat/stage1" --base "$BASE" --level topology
    printf '%q ' --out "${DATA%.jsonl}_flat_scd.json" --batch-size "$EVAL_BATCH" --max-new 1400
    printf '%s\n' --dry-run
    printf '+ %q ' "$PY" -u -m eval.eval_constrained --data "$DATA"
    printf '%q ' --adapter "$ADAPTER_ROOT/spcl/stage3" --base "$BASE" --level topology
    printf '%q ' --out "${DATA%.jsonl}_spcl_scd.json" --batch-size "$EVAL_BATCH" --max-new 1400
    printf '%s\n' --dry-run
  fi
else
  run "$PY" -u -m experiments.revision.evaluate_external_domain "${EVAL_ARGS[@]}" \
    --adapter "$ADAPTER_ROOT/flat/stage1" --out "${DATA%.jsonl}_flat.json"
  run "$PY" -u -m experiments.revision.evaluate_external_domain "${EVAL_ARGS[@]}" \
    --adapter "$ADAPTER_ROOT/spcl/stage3" --out "${DATA%.jsonl}_spcl.json"
  if [[ "$RUN_SCD" == 1 ]]; then
    run "$PY" -u -m eval.eval_constrained --data "$DATA" \
      --adapter "$ADAPTER_ROOT/flat/stage1" --base "$BASE" --level topology \
      --out "${DATA%.jsonl}_flat_scd.json" --batch-size "$EVAL_BATCH" --max-new 1400
    run "$PY" -u -m eval.eval_constrained --data "$DATA" \
      --adapter "$ADAPTER_ROOT/spcl/stage3" --base "$BASE" --level topology \
      --out "${DATA%.jsonl}_spcl_scd.json" --batch-size "$EVAL_BATCH" --max-new 1400
  fi
fi
echo "External-domain protocol emitted. DRY_RUN=$DRY_RUN RUN_SCD=$RUN_SCD DATA=$DATA"
