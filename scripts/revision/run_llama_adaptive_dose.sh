#!/usr/bin/env bash
# Fully state-derived SPCL protocol for Llama-3.2-1B.
# Round 1 is uniform over the easy prefix.  At each later boundary the
# builder derives relative bucket dose from Dbar_b * R_b, where R_b is the
# training-only residual completion NLL measured at the current checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/root/miniconda3/bin/python}"
BASE="${BASE:-models/llama32-1b}"
BACKBONE_NAME="${BACKBONE_NAME:-llama32_1b_adaptive_dose}"
SIDECAR="${SIDECAR:-outputs/revision/attribution_v2/seed42/curriculum/spcl/spcl_scores.jsonl}"
SEED="${SEED:-42}"
ROOT="${ROOT:-outputs/revision/${BACKBONE_NAME}/seed${SEED}}"
TRAIN="${TRAIN:-data/train_aug10.jsonl}"
VAL="${VAL:-data/val.jsonl}"
TEST="${TEST:-data/test.jsonl}"
DRY_RUN="${DRY_RUN:-1}"
BATCH="${BATCH:-64}"
ACCUM="${ACCUM:-1}"
MAX_LEN="${MAX_LEN:-2048}"
EVAL_BATCH="${EVAL_BATCH:-8}"
PROBE_BATCH="${PROBE_BATCH:-4}"
LOSS_TYPE="${LOSS_TYPE:-chunked_nll}"
run() { printf '+ '; printf '%q ' "$@"; printf '\n'; [[ "$DRY_RUN" == 1 ]] || "$@"; }
[[ -f "$SIDECAR" && -d "$BASE" ]] || { echo "missing BASE or SIDECAR" >&2; exit 2; }
CUR="$ROOT/curriculum/adaptive"; CKPT="$ROOT/checkpoints"; PROBES="$ROOT/probes"; EVAL="$ROOT/eval"
mkdir -p "$CUR" "$PROBES" "$EVAL"
MANIFEST="$ROOT/manifest.json"
run "$PY" -u -m experiments.revision.protocol manifest --phase "backbone_adaptive/${BACKBONE_NAME}" --status planned \
  --base "$BASE" --seed "$SEED" --train "$TRAIN" --val "$VAL" --test "$TEST" \
  --optimizer-steps "{\"flat_epochs\":0,\"spcl_rounds\":3,\"effective_batch\":$((BATCH*ACCUM)),\"micro_batch\":$BATCH,\"gradient_accumulation\":$ACCUM,\"loss_type\":\"$LOSS_TYPE\",\"max_len\":$MAX_LEN,\"dose_rule\":\"m_b proportional to Dbar_b times R_b; R_b training NLL residual; no grid/floor/cap\"}" \
  --decode-config "{\"greedy\":true,\"max_new_tokens\":1400,\"eval_batch\":$EVAL_BATCH}" --out "$MANIFEST" --overwrite

# Base NLL is the denominator for all subsequent residual measurements.
BASE_REPORT="$PROBES/base_nll.json"
run "$PY" -u -m curriculum.residual_probe --mode score --train "$TRAIN" --scores "$SIDECAR" --base "$BASE" \
  --label "${BACKBONE_NAME}_base" --device cuda --batch-size "$PROBE_BATCH" --max-len "$MAX_LEN" --out "$BASE_REPORT"
PREV=""
for ROUND in 1 2 3; do
  IDX=$((ROUND-1)); EXTRA=()
  if [[ "$ROUND" == 1 ]]; then
    run "$PY" -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" --out-dir "$CUR" --rounds 3 --buckets 4 --window 2,3,4 \
      --scores-cache "$SIDECAR" --only-round "$IDX" --base "$BASE" --seed "$SEED"
  else
    PREV_REPORT="$PROBES/s${ROUND}_nll.json"; GATE="$PROBES/gate_s$((ROUND-1)).json"
    run "$PY" -u -m curriculum.residual_probe --mode gate --base-report "$BASE_REPORT" --report "$PROBES/s$((ROUND-1))_nll.json" --out "$GATE"
    run "$PY" -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" --out-dir "$CUR" --rounds 3 --buckets 4 --window 2,3,4 \
      --scores-cache "$SIDECAR" --only-round "$IDX" --base "$BASE" --seed "$SEED" --adaptive-dose-residual "$GATE"
  fi
  INIT=(); [[ -z "$PREV" ]] || INIT=(--init-adapter "$PREV")
  run "$PY" -u -m training.sft_lora --mode flat --stages "$CUR/round${ROUND}.jsonl" --val "$VAL" --run-name "adaptive_r${ROUND}" \
    --checkpoint-root "$CKPT" --base "$BASE" --epochs 1 --lr 1e-4 --batch "$BATCH" --accum "$ACCUM" --max-len "$MAX_LEN" \
    --loss-type "$LOSS_TYPE" --seed "$((SEED+ROUND-1))" "${INIT[@]}"
  PREV="$CKPT/adaptive_r${ROUND}/stage1"
  if [[ "$ROUND" -lt 3 ]]; then
    run "$PY" -u -m curriculum.residual_probe --mode score --train "$TRAIN" --scores "$SIDECAR" --base "$BASE" --adapter "$PREV" \
      --label "${BACKBONE_NAME}_s${ROUND}" --device cuda --batch-size "$PROBE_BATCH" --max-len "$MAX_LEN" --out "$PROBES/s${ROUND}_nll.json"
  fi
done
run "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" --adapter "$PREV" --base "$BASE" --out "$EVAL/spcl_adaptive_test.json" --batch-size "$EVAL_BATCH" --max-new 1400
run "$PY" -u -m eval.eval_constrained --data "$TEST" --adapter "$PREV" --base "$BASE" --level topology --out "$EVAL/spcl_adaptive_scd_test.json" --batch-size "$EVAL_BATCH" --max-new 1400
echo "Adaptive-dose Llama protocol emitted. DRY_RUN=$DRY_RUN ROOT=$ROOT"
