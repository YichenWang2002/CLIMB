#!/usr/bin/env bash
# Three-seed mechanism replication: SPCL, mixs, hardtail, anti.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/root/miniconda3/bin/python}"
BASE="${BASE:-models/llama32-1b}"
SEEDS="${SEEDS:-42 43 44}"
ARMS="${ARMS:-spcl mixs hardtail anti}"
ROOT="${ROOT:-outputs/revision/attribution}"
TRAIN="${TRAIN:-data/train_aug10.jsonl}"
VAL="${VAL:-data/val.jsonl}"
TEST="${TEST:-data/test.jsonl}"
DRY_RUN="${DRY_RUN:-1}"
MAX_LEN="${MAX_LEN:-2048}"

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}

for SEED in $SEEDS; do
  SEED_ROOT="$ROOT/seed${SEED}"
  SOURCE="$SEED_ROOT/curriculum/spcl"
  CKPT="$SEED_ROOT/checkpoints"
  EVAL="$SEED_ROOT/eval"
  if [[ -e "$SEED_ROOT/manifest.json" && "$DRY_RUN" != 1 ]]; then
    echo "refusing to reuse existing attribution run: $SEED_ROOT" >&2
    exit 3
  fi
  mkdir -p "$SEED_ROOT"
  run "$PY" -u -m experiments.revision.protocol manifest \
    --phase attribution --status planned --base "$BASE" --seed "$SEED" \
    --train "$TRAIN" --val "$VAL" --test "$TEST" \
    --optimizer-steps '{"spcl_rounds":3,"control_rounds":3,"effective_batch":16}' \
    --decode-config '{"greedy":true,"max_new_tokens":1400}' \
    --out "$SEED_ROOT/manifest.json" --overwrite
  if [[ ! -f "$SOURCE/report.json" ]]; then
    run "$PY" -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" \
        --out-dir "$SOURCE" --rounds 3 --buckets 4 --window 2,3,4 \
        --boosts '1,1|1,1,1.5|0.5,0.75,1.5,3' --batch-size 8 --max-len "$MAX_LEN" \
      --embed-device cpu --base "$BASE" --seed "$SEED"
  fi
  for ARM in $ARMS; do
    CUR="$SOURCE"
    if [[ "$ARM" != spcl ]]; then
      CUR="$SEED_ROOT/curriculum/$ARM"
      run "$PY" -u -m experiments.revision.build_attribution --method "$ARM" \
        --train "$TRAIN" --spcl-dir "$SOURCE" --out-dir "$CUR" --seed "$SEED"
    fi
    if [[ "$ARM" == spcl ]]; then
      STAGES=("$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl")
      EPOCHS=1
    else
      STAGES=("$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl")
      EPOCHS=1
    fi
    run "$PY" -u -m training.sft_lora --mode staged --stages "${STAGES[@]}" \
      --val "$VAL" --run-name "$ARM" --checkpoint-root "$CKPT" --base "$BASE" \
      --epochs "$EPOCHS" --lr 1e-4 --batch 4 --accum 4 --max-len "$MAX_LEN" \
      --loss-type chunked_nll --seed "$SEED" --preserve-order
    run "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
      --adapter "$CKPT/$ARM/stage3" --base "$BASE" --out "$EVAL/${ARM}_test.json" \
      --batch-size 32 --max-new 1400
  done
done

run "$PY" -u -m experiments.revision.aggregate_revision --root "$ROOT" \
  --baseline spcl --out "$ROOT/aggregate.json" --bootstrap-samples 10000
if [[ "$DRY_RUN" != 1 ]]; then
  "$PY" -u -m experiments.revision.protocol manifest \
    --phase attribution --status complete --base "$BASE" \
    --train "$TRAIN" --val "$VAL" --test "$TEST" \
    --out "$ROOT/completed_manifest.json" --overwrite
fi
echo "Attribution protocol emitted. DRY_RUN=$DRY_RUN ROOT=$ROOT"
