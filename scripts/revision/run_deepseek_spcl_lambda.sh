#!/usr/bin/env bash
# Reproducible DeepSeek-R1-Distill-Qwen-1.5B replication of SPCL-lambda.
#
# All curriculum decisions are made from the training split.  The semantic
# difficulty/utility sidecar is scored once with this backbone; at each round
# boundary, residual_probe computes R_b from teacher-forced completion NLL and
# build_spcl applies m_realized = 1 + (m_nominal - 1) R_b for m_nominal > 1.
# Flat SFT and SPCL use the same data, LoRA recipe, optimizer budget and test
# decoder.  Test is touched only by the final strict evaluation commands.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/root/miniconda3/bin/python}"

BASE="${BASE:-../model/DeepSeek-R1-Distill-Qwen-1.5B}"
NAME="${NAME:-deepseek_r1_qwen15b_spcl_lambda}"
ROOT="${ROOT:-outputs/revision/${NAME}}"
SEED="${SEED:-42}"
TRAIN="${TRAIN:-data/train_aug10.jsonl}"
VAL="${VAL:-data/val.jsonl}"
TEST="${TEST:-data/test.jsonl}"
BATCH="${BATCH:-16}"
ACCUM="${ACCUM:-1}"
MAX_LEN="${MAX_LEN:-2048}"
# Large-vocabulary generation is stable at 64 on the 24GB GPU; use it by
# default so strict evaluation does not under-utilize the available device.
EVAL_BATCH="${EVAL_BATCH:-64}"
LOSS_TYPE="${LOSS_TYPE:-chunked_nll}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SCD="${RUN_SCD:-1}"
BOOSTS="${BOOSTS:-1,1|1,1,1.5|0.5,0.75,1.5,3}"

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}

[[ -d "$BASE" && -f "$TRAIN" && -f "$VAL" && -f "$TEST" ]] || {
  echo "missing BASE or dataset split" >&2; exit 2;
}
if [[ -e "$ROOT/manifest.json" && "$DRY_RUN" != 1 && "${OVERWRITE:-0}" != 1 ]]; then
  echo "refusing to reuse existing run: $ROOT (choose a new ROOT)" >&2; exit 3
fi

SEED_ROOT="$ROOT/seed${SEED}"
CUR="$SEED_ROOT/curriculum/spcl_lambda"
CKPT="$SEED_ROOT/checkpoints"
EVAL="$SEED_ROOT/eval"
PROBE="$SEED_ROOT/probes"
mkdir -p "$SEED_ROOT" "$PROBE" "$EVAL"

run "$PY" -u -m experiments.revision.protocol manifest \
  --phase "backbone_lambda/${NAME}" --status planned --hash-weights \
  --base "$BASE" --seed "$SEED" --train "$TRAIN" --val "$VAL" --test "$TEST" \
  --optimizer-steps "{\"flat_epochs\":3,\"spcl_rounds\":3,\"effective_batch\":$((BATCH*ACCUM)),\"micro_batch\":$BATCH,\"gradient_accumulation\":$ACCUM,\"loss_type\":\"$LOSS_TYPE\",\"max_len\":$MAX_LEN,\"dose_rule\":\"NLL residual R_b; no validation/test\"}" \
  --decode-config "{\"greedy\":true,\"max_new_tokens\":1400,\"eval_batch\":$EVAL_BATCH}" \
  --out "$SEED_ROOT/manifest.json" --overwrite

echo "=== score DeepSeek difficulty/utility and build round 1 ==="
run "$PY" -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" \
  --out-dir "$CUR" --rounds 3 --buckets 4 --window 2,3,4 \
  --boosts "$BOOSTS" --batch-size "$BATCH" --max-len "$MAX_LEN" \
  --embed-device cpu --base "$BASE" --seed "$SEED" --only-round 0

SIDECAR="$CUR/spcl_scores.jsonl"
echo "=== frozen-base NLL report (training only) ==="
run "$PY" -u -m curriculum.residual_probe --mode score --train "$TRAIN" \
  --scores "$SIDECAR" --base "$BASE" --label "${NAME}_base" --device cuda \
  --batch-size "$BATCH" --max-len "$MAX_LEN" --out "$PROBE/base_nll.json"

echo "=== flat SFT (3 epochs, budget matched) ==="
run "$PY" -u -m training.sft_lora --mode flat --stages "$TRAIN" --val "$VAL" \
  --run-name flat --checkpoint-root "$CKPT" --base "$BASE" --epochs 3 \
  --lr 1e-4 --batch "$BATCH" --accum "$ACCUM" --max-len "$MAX_LEN" \
  --loss-type "$LOSS_TYPE" --seed "$SEED"

PREV=""
for ROUND in 1 2 3; do
  if [[ "$ROUND" -gt 1 ]]; then
    PREV_IDX=$((ROUND-1))
    echo "=== NLL probe theta_${PREV_IDX} and parameter-free gate ==="
    run "$PY" -u -m curriculum.residual_probe --mode score --train "$TRAIN" \
      --scores "$SIDECAR" --base "$BASE" --adapter "$PREV" \
      --label "${NAME}_theta${PREV_IDX}" --device cuda --batch-size "$BATCH" \
      --max-len "$MAX_LEN" --out "$PROBE/theta${PREV_IDX}_nll.json"
    run "$PY" -u -m curriculum.residual_probe --mode gate \
      --base-report "$PROBE/base_nll.json" --report "$PROBE/theta${PREV_IDX}_nll.json" \
      --boosts "$BOOSTS" --label "${NAME}_theta${PREV_IDX}_gate" \
      --out "$PROBE/theta${PREV_IDX}_gate.json"
  fi
  IDX=$((ROUND-1))
  GATE=()
  if [[ "$ROUND" -gt 1 ]]; then GATE=(--gate-residual "$PROBE/theta$((ROUND-1))_gate.json"); fi
  echo "=== build adaptive SPCL round ${ROUND} ==="
  run "$PY" -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" \
    --out-dir "$CUR" --rounds 3 --buckets 4 --window 2,3,4 --boosts "$BOOSTS" \
    --scores-cache "$SIDECAR" --only-round "$IDX" --base "$BASE" --seed "$SEED" \
    "${GATE[@]}"
  echo "=== train adaptive SPCL round ${ROUND} ==="
  INIT=(); if [[ -n "$PREV" ]]; then INIT=(--init-adapter "$PREV"); fi
  run "$PY" -u -m training.sft_lora --mode flat --stages "$CUR/round${ROUND}.jsonl" \
    --val "$VAL" --run-name "spcl_r${ROUND}" --checkpoint-root "$CKPT" --base "$BASE" \
    --epochs 1 --lr 1e-4 --batch "$BATCH" --accum "$ACCUM" --max-len "$MAX_LEN" \
    --loss-type "$LOSS_TYPE" --seed "$((SEED+ROUND-1))" "${INIT[@]}"
  PREV="$CKPT/spcl_r${ROUND}/stage1"
done

echo "=== strict held-out evaluation (single final touch) ==="
run "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
  --adapter "$CKPT/flat/stage1" --base "$BASE" --out "$EVAL/flat_test.json" \
  --batch-size "$EVAL_BATCH" --max-new 1400
run "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
  --adapter "$CKPT/spcl_r3/stage1" --base "$BASE" --out "$EVAL/spcl_lambda_test.json" \
  --batch-size "$EVAL_BATCH" --max-new 1400
if [[ "$RUN_SCD" == 1 ]]; then
  run "$PY" -u -m eval.eval_constrained --data "$TEST" --adapter "$CKPT/flat/stage1" \
    --base "$BASE" --level topology --out "$EVAL/flat_scd_test.json" \
    --batch-size "$EVAL_BATCH" --max-new 1400
  run "$PY" -u -m eval.eval_constrained --data "$TEST" --adapter "$CKPT/spcl_r3/stage1" \
    --base "$BASE" --level topology --out "$EVAL/spcl_lambda_scd_test.json" \
    --batch-size "$EVAL_BATCH" --max-new 1400
fi
run "$PY" -u -m experiments.revision.aggregate_revision --root "$ROOT" \
  --baseline flat --out "$ROOT/aggregate.json" --bootstrap-samples 10000
echo "DeepSeek SPCL-lambda complete: $ROOT"
