#!/usr/bin/env bash
# Frozen-checkpoint harmonization audit for historical Llama runs.
# Inference only: current strict XML/schema/executor evaluator, no retraining.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="${PY:-/root/miniconda3/bin/python}"
BASE="${BASE:-models/llama32-1b}"
TEST="${TEST:-data/test.jsonl}"
ROOT="${ROOT:-outputs/revision/llama32_1b_strict_audit}"
SEEDS="${SEEDS:-42 43 44 45 46}"
EVAL_BATCH="${EVAL_BATCH:-64}"
MAX_NEW="${MAX_NEW:-1400}"
DRY_RUN="${DRY_RUN:-1}"

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}

[[ -f "$TEST" ]] || { echo "missing TEST=$TEST" >&2; exit 2; }
[[ -d "$BASE" ]] || { echo "missing BASE=$BASE" >&2; exit 2; }

for SEED in $SEEDS; do
  case "$SEED" in
    42) FLAT="outputs/checkpoints/spcl_cmp_flat/stage1"; SPCL="outputs/checkpoints/spcl_cmp_method_v2/stage3" ;;
    *)  FLAT="outputs/checkpoints/flat_s${SEED}/stage1"; SPCL="outputs/checkpoints/spcl_v2_s${SEED}/stage3" ;;
  esac
  OUT="$ROOT/seed${SEED}/eval"
  mkdir -p "$OUT"
  [[ -d "$FLAT" && -d "$SPCL" ]] || {
    echo "missing frozen adapters for seed $SEED: flat=$FLAT spcl=$SPCL" >&2; exit 2;
  }
  if [[ "${OVERWRITE:-0}" != 1 && ( -e "$OUT/flat_test.json" || -e "$OUT/spcl_test.json" ) ]]; then
    echo "existing audit output under $OUT; choose another ROOT or set OVERWRITE=1" >&2
    exit 3
  fi
  run "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$FLAT" --base "$BASE" --out "$OUT/flat_test.json" \
    --batch-size "$EVAL_BATCH" --max-new "$MAX_NEW"
  run "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$SPCL" --base "$BASE" --out "$OUT/spcl_test.json" \
    --batch-size "$EVAL_BATCH" --max-new "$MAX_NEW"
  run "$PY" -u -m experiments.revision.report_revision \
    --run flat="$OUT/flat_test.json" --run spcl="$OUT/spcl_test.json" \
    --out "$OUT/strict_report.json" --bootstrap-samples 10000
  if [[ "$DRY_RUN" != 1 ]]; then
    "$PY" - "$ROOT/seed${SEED}/audit_manifest.json" <<PY
import json, sys
from pathlib import Path
payload = {
    "audit_type": "frozen_checkpoint_harmonization",
    "seed": int("$SEED"),
    "base": "$BASE",
    "test": "$TEST",
    "flat_adapter": "$FLAT",
    "spcl_adapter": "$SPCL",
    "evaluator": "experiments.bt_ducl.strict_eval",
    "decoding": {"greedy": True, "max_new_tokens": int("$MAX_NEW"),
                  "batch_size": int("$EVAL_BATCH")},
    "training": "not run; historical frozen checkpoints",
}
target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  fi
done

echo "Llama strict harmonization audit emitted. DRY_RUN=$DRY_RUN ROOT=$ROOT"

