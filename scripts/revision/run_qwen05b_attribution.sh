#!/usr/bin/env bash
# Qwen2.5-0.5B attribution controls under the same strict evaluator as SPCL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/root/miniconda3/bin/python}"
BASE="${BASE:-../qwen25-05B}"
SEED="${SEED:-42}"
ARMS="${ARMS:-hardtail anti}"
ROOT="${ROOT:-outputs/revision/qwen25_05b_window123/seed${SEED}}"
TRAIN="${TRAIN:-data/train_aug10.jsonl}"
VAL="${VAL:-data/val.jsonl}"
TEST="${TEST:-data/test.jsonl}"
SOURCE="$ROOT/curriculum/spcl_b1"
WINDOWS="${WINDOWS:-1,2,3}"
BUCKETS="${BUCKETS:-3}"
BOOSTS="${BOOSTS:-1|1,1|0.75,1.5,2.0}"

for ARM in $ARMS; do
  CUR="$ROOT/curriculum/$ARM"
  CKPT="$ROOT/checkpoints/$ARM"
  EVAL="$ROOT/eval/${ARM}_test.json"
  if [[ "$ARM" == "mixs" ]]; then
    "$PY" -u -m experiments.revision.build_attribution --method mixs \
      --train "$TRAIN" --spcl-dir "$SOURCE" --out-dir "$CUR" --seed "$SEED" \
      --rounds 3 --buckets "$BUCKETS" --windows "$WINDOWS" --boosts "$BOOSTS"
  elif [[ ! -f "$CUR/report.json" ]]; then
    "$PY" -u -m experiments.revision.build_attribution --method "$ARM" \
      --train "$TRAIN" --spcl-dir "$SOURCE" --out-dir "$CUR" --seed "$SEED" \
      --rounds 3 --buckets "$BUCKETS" --windows "$WINDOWS" --boosts "$BOOSTS"
  fi
  "$PY" -u -m training.sft_lora --mode staged \
    --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
    --val "$VAL" --run-name "$ARM" --checkpoint-root "$ROOT/checkpoints" --base "$BASE" \
    --epochs 1 --lr 1e-4 --batch 16 --accum 1 --max-len 2048 \
    --loss-type chunked_nll --seed "$SEED"
  "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$CKPT/stage3" --base "$BASE" --out "$EVAL" \
    --batch-size 64 --max-new 1400
done

echo "Completed Qwen0.5B controls: $ARMS"
