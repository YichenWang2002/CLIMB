#!/usr/bin/env bash
# Post-hoc topology-grounded decoding for already trained adapters.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/root/miniconda3/bin/python}"
BASE="${BASE:-../model/qwen25-15b}"
ROOT="${ROOT:-outputs/revision/qwen25_1p5b_chunked_b16_rerun}"
TEST="${TEST:-data/test.jsonl}"
SEEDS="${SEEDS:-42}"
EVAL_BATCH="${EVAL_BATCH:-4}"
MAX_NEW="${MAX_NEW:-1400}"
DRY_RUN="${DRY_RUN:-1}"

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}

[[ -f "$TEST" ]] || { echo "missing TEST=$TEST" >&2; exit 2; }
[[ -d "$BASE" ]] || { echo "missing BASE=$BASE" >&2; exit 2; }

for SEED in $SEEDS; do
  ADAPTERS="$ROOT/seed${SEED}/checkpoints"
  OUT="$ROOT/seed${SEED}/eval"
  [[ -d "$ADAPTERS/flat/stage1" && -d "$ADAPTERS/spcl/stage3" ]] || {
    echo "trained flat/SPCL adapters not found under $ADAPTERS" >&2; exit 2;
  }
  mkdir -p "$OUT"
  run "$PY" -u -m eval.eval_constrained --data "$TEST" \
    --adapter "$ADAPTERS/flat/stage1" --base "$BASE" --level topology \
    --out "$OUT/flat_scd_test.json" --batch-size "$EVAL_BATCH" --max-new "$MAX_NEW"
  run "$PY" -u -m eval.eval_constrained --data "$TEST" \
    --adapter "$ADAPTERS/spcl/stage3" --base "$BASE" --level topology \
    --out "$OUT/spcl_scd_test.json" --batch-size "$EVAL_BATCH" --max-new "$MAX_NEW"
done
echo "Post-hoc SCD evaluation emitted. DRY_RUN=$DRY_RUN ROOT=$ROOT"
